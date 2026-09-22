#!/usr/bin/env python3
"""Post-run official-tokenizer oracle for fixed-text FPGA model traces.

Host tokenization/decoding only verifies captured board results. It never feeds
model inputs or participates in the board run. Numeric model acceptance remains
separate. Raw output is parsed by announced byte length, not marker searching.
"""
import argparse
import codecs
import hashlib
import json
from pathlib import Path
import re

from check_kernel_profile import STAGE

H8 = rb'[0-9A-Fa-f]{8}'
H16 = rb'[0-9A-Fa-f]{16}'
PROMPT = re.compile(rb'\[text\] prompt count=(' + H8 + rb') ids=(' + H8 +
                    rb'(?: ' + H8 + rb')*) encode_cycles=(' + H16 + rb')')
PREDICTION = re.compile(rb'\[text\] prediction=(' + H8 + rb') token=(' + H8 +
                        rb') eos=(' + H8 + rb') bytes=(' + H8 + rb') hex=([0-9a-f]*)')
OUTPUT = re.compile(rb'\[text\] output bytes=(' + H8 + rb') hex=([0-9a-f]*)')
BEGIN = re.compile(rb'\[model\] (prefill|decode) begin position=(' + H8 +
                   rb') input_token=(' + H8 + rb')')
MODE = b'[text] mode=fixed-validation; EOS is recorded, never early-stops'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def from_hex(value):
    require(len(value) % 2 == 0, 'odd-length UTF-8 hex field')
    return bytes.fromhex(value.decode('ascii'))


class OfficialOracle:
    """Official template/tokenization plus ByteLevel incremental UTF-8 decoding."""
    def __init__(self, assets):
        from transformers import AutoTokenizer
        self.assets = assets
        self.tokenizer = AutoTokenizer.from_pretrained(str(assets), local_files_only=True)
        tokenizer_path = assets/'tokenizer.json'
        self.config = json.loads(tokenizer_path.read_text())
        require(self.config.get('decoder', {}).get('type') == 'ByteLevel',
                'unsupported official decoder: expected ByteLevel')
        generation = json.loads((assets/'generation_config.json').read_text())
        eos = generation.get('eos_token_id')
        self.eos = set(eos if isinstance(eos, list) else [eos])
        require(self.eos == {151643, 151645}, 'official EOS config differs from current firmware contract')
        self.special = {x['id'] for x in self.config['added_tokens'] if x['special']}
        self.added = {x['id']: x for x in self.config['added_tokens']}
        byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        codepoints = list(byte_values)
        extra = 0
        for b in range(256):
            if b not in byte_values:
                byte_values.append(b)
                codepoints.append(256 + extra)
                extra += 1
        # Construct the standard GPT-2/Qwen ByteLevel alphabet explicitly. The
        # 188 initially visible byte values retain their original codepoints.
        self.byte_inverse = {chr(c): b for c, b in zip(codepoints, byte_values)}
        self.hashes = {str(p): digest(p.read_bytes()) for p in
                       (tokenizer_path, assets/'tokenizer_config.json', assets/'generation_config.json')}

    def prompt_ids(self, text, thinking):
        rendered = self.tokenizer.apply_chat_template([{'role':'user','content':text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return self.tokenizer(rendered, add_special_tokens=False)['input_ids']

    def incremental_decode(self, tokens):
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        chunks = []
        for token in tokens:
            if token in self.special:
                raw = b''
            else:
                piece = self.tokenizer.convert_ids_to_tokens(token)
                require(isinstance(piece, str), 'unknown predicted token ID')
                if token in self.added:
                    # Non-special added tokens are not used by this checkpoint;
                    # reject them instead of guessing how its decoder handles them.
                    raise ValueError('unsupported non-special added token in prediction')
                require(all(c in self.byte_inverse for c in piece), 'non-ByteLevel vocabulary item')
                raw = bytes(self.byte_inverse[c] for c in piece)
            chunks.append(decoder.decode(raw, final=False).encode('utf-8'))
        finish = decoder.decode(b'', final=True).encode('utf-8')
        combined = b''.join(chunks) + finish
        official = self.tokenizer.decode(tokens, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False).encode('utf-8')
        require(combined == official, 'incremental ByteLevel oracle differs from official complete decoder')
        return chunks, finish, official


def framed_lines(raw):
    """Exclude raw model text from parsing diagnostic/control markers."""
    records, position, announced = [], 0, None
    while position < len(raw):
        end = raw.find(b'\n', position)
        require(end >= 0, 'truncated final UART diagnostic line')
        line = raw[position:end]
        if line.endswith(b'\r'):
            line = line[:-1]
        position = end + 1
        records.append(('line', line))
        output = OUTPUT.fullmatch(line)
        if output:
            require(announced is None, 'duplicate final output announcement')
            announced = (int(output[1], 16), from_hex(output[2]))
            require(announced[0] == len(announced[1]), 'final output byte count differs from hex')
        if line == b'[text] BEGIN':
            require(announced is not None, 'raw text begins before length announcement')
            require(len(records) >= 2 and OUTPUT.fullmatch(records[-2][1]),
                    'raw text frame must immediately follow output announcement')
            length, expected = announced
            body = raw[position:position + length]
            require(len(body) == length and body == expected, 'raw framed output differs from declared exact bytes')
            position += length
            delimiter = b'\r\n[text] END\r\n'
            require(raw[position:position+len(delimiter)] == delimiter, 'bad/truncated raw output frame delimiter')
            position += len(delimiter)
            records.append(('raw', body))
            records.append(('line', b'[text] END'))
    return records


def verify(raw, plan, oracle):
    fixed = plan.get('fixed_prompt')
    require(plan.get('text_mode') == 'fixed-validation' and isinstance(fixed, dict), 'image is not fixed-text validation')
    prompt = bytes.fromhex(fixed['utf8_hex'])
    require(prompt and len(prompt) == fixed['bytes'] and digest(prompt) == fixed['sha256'], 'fixed prompt identity mismatch')
    text = prompt.decode('utf-8', errors='strict')
    require(fixed.get('thinking') is False, 'current firmware requires thinking=false')
    require(fixed.get('prefill_calls') == 1 and fixed.get('decode_calls') == 8
            and fixed.get('predictions_recorded') == 9, 'requires exactly one prefill plus eight decode calls')
    expected_ids = oracle.prompt_ids(text, fixed['thinking'])
    require(len(expected_ids) == 16 and expected_ids == fixed['expected_ids'], 'official prompt IDs differ from image expected IDs')
    records = framed_lines(raw)
    events, predictions, models, starts = [], [], [], []
    prompt_row, finish, output, payload = None, None, None, None
    nr_pass, runtime_pass, text_pass, tokenizer_pass = 0, 0, 0, 0
    for kind, line in records:
        if kind == 'raw':
            payload = line
            events.append('raw')
            continue
        if re.search(rb'\bFAIL\b|TRAP mcause=|\[nr\].*\bTRAP\b', line):
            raise ValueError('firmware failure marker outside raw output')
        launch = re.fullmatch(rb'\[nr\] launch cycles=0x[0-9A-Fa-f]+ status=0x([0-9A-Fa-f]+)',line)
        if launch:
            require(int(launch[1],16) == 0, 'nonzero runtime status')
        if m := PROMPT.fullmatch(line):
            count, values, cycles = m.groups()
            prompt_row = {'ids':[int(x,16) for x in values.split()], 'count':int(count,16),
                          'encode_cycles':int(cycles,16)}
            require(prompt_row['count'] == 16 and prompt_row['ids'] == expected_ids
                    and prompt_row['encode_cycles'] > 0, 'actual FPGA tokenizer IDs/count/cycles differ from official oracle')
            events.append('prompt')
        elif line == b'verify fixed prompt tokenizer: PASS':
            tokenizer_pass += 1; events.append('tokenizer_pass')
        elif line == MODE:
            events.append('mode')
        elif m := BEGIN.fullmatch(line):
            starts.append({'kind':m[1].decode(),'position':int(m[2],16),'input_token':int(m[3],16)})
            events.append('begin')
        elif m := STAGE.fullmatch(line.decode('ascii',errors='replace')):
            graph, pos, token, bits, cycles = m.groups()
            require(int(cycles,16) > 0, 'model compute cycles must be positive')
            models.append({'kind':graph,'position':int(pos,16),'token':int(token,16)})
            events.append('model')
        elif m := PREDICTION.fullmatch(line):
            index, token, eos, size, value = m.groups(); chunk = from_hex(value)
            row = {'prediction':int(index,16),'token':int(token,16),'eos':int(eos,16),
                   'bytes':int(size,16),'hex':chunk.hex()}
            require(row['bytes'] == len(chunk), 'prediction emitted-byte count mismatch')
            require(row['eos'] == int(row['token'] in oracle.eos), 'prediction EOS marker differs from official config')
            predictions.append(row); events.append('prediction')
        elif m := re.fullmatch(rb'\[text\] finish hex=([0-9a-f]*)',line):
            finish = from_hex(m[1]); events.append('finish')
        elif m := OUTPUT.fullmatch(line):
            output = from_hex(m[2]); events.append('output')
        elif line == b'[text] BEGIN':
            events.append('frame_begin')
        elif line == b'[text] END':
            events.append('frame_end')
        elif line == b'verify fixed text validation: PASS':
            text_pass += 1; events.append('text_pass')
        elif line == b'[nr] RA returned: PASS':
            nr_pass += 1; events.append('nr_pass')
        elif line == b'verify NR runtime: PASS':
            runtime_pass += 1; events.append('runtime_pass')
        elif (b'[text]' in line or line.startswith(b'verify fixed ')
              or line.startswith(b'[nr] RA returned:') or line.startswith(b'verify NR runtime:')
              or re.match(rb'\[model\] (?:prefill|decode)\b',line)):
            raise ValueError('unknown/malformed text or model control record: '+line[:120].decode(errors='replace'))
    expected_events = ['prompt','tokenizer_pass','mode'] + ['begin','model','prediction']*9 + [
        'finish','output','frame_begin','raw','frame_end','text_pass','nr_pass','runtime_pass']
    require(events == expected_events, 'missing/duplicate/out-of-order fixed-text records or runtime completion')
    require(nr_pass == runtime_pass == text_pass == tokenizer_pass == 1, 'exactly one completion of each kind required')
    expected_stages = [('prefill',0)] + [('decode',p) for p in range(16,24)]
    require([(r['kind'],r['position']) for r in models] == expected_stages
            and [(r['kind'],r['position']) for r in starts] == expected_stages,
            'prefill/decode positions must be 0 then 16..23')
    require([r['prediction'] for r in predictions] == list(range(9)), 'prediction indices must be 0..8')
    tokens = [r['token'] for r in predictions]
    require(tokens == [r['token'] for r in models], 'text predictions differ from actual model-selected tokens')
    require([r['input_token'] for r in starts] == [expected_ids[0]] + tokens[:-1], 'decode input token trajectory mismatch')
    chunks, wanted_finish, wanted_output = oracle.incremental_decode(tokens)
    require([r['hex'] for r in predictions] == [c.hex() for c in chunks], 'incremental emitted UTF-8 differs from official oracle')
    require(finish == wanted_finish, 'final UTF-8 flush differs from official oracle')
    require(output == payload == wanted_output, 'complete decoded UTF-8 differs from official oracle')
    require(len(output) <= fixed['output_capacity'], 'firmware output exceeds image capacity')
    return {'status':'FIXED_TEXT_PASS','execution':'FPGA trace verified post-run on host',
            'scope':'fixed raw prompt -> board template/tokenize -> actual graph-selected predictions -> board incremental UTF-8 decode',
            'prompt_utf8_hex':prompt.hex(),'actual_prompt':prompt_row,'official_prompt_ids':expected_ids,
            'model_stages':models,'predictions':predictions,'finish_hex':finish.hex(),
            'output_utf8_hex':output.hex(),'output_text':output.decode('utf-8'),
            'limits':['Fixed prompt fixture only; no UART RX or arbitrary user interaction acceptance.',
                      'Text validation does not replace numerical logits/KV or hidden-state verification.',
                      'Nine predictions mean one prefill-selected token plus eight sequential decode predictions.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('uart','image-plan','assets','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    raw = args.uart.read_bytes()
    try:
        oracle = OfficialOracle(args.assets)
        report = verify(raw,json.loads(args.image_plan.read_text()),oracle)
        report['official_assets_sha256'] = oracle.hashes
    except (ValueError,KeyError,TypeError,UnicodeError) as error:
        report = {'status':'NOT_ACCEPTED','errors':[str(error)],'scope':'post-run fixed-text verification'}
    report['inputs_sha256'] = {str(args.uart):digest(raw),str(args.image_plan):digest(args.image_plan.read_bytes())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':report['status'],'output':str(args.output)}))
    return 0 if report['status']=='FIXED_TEXT_PASS' else 1


if __name__=='__main__':raise SystemExit(main())
