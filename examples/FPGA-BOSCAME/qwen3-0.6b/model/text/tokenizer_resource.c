#include "tokenizer_resource.h"

static uint32_t u32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
         (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static int range(size_t offset, size_t count, size_t size) {
  return offset <= size && count <= size - offset;
}
int qwen_tokenizer_open(QwenTokenizerResource *out, const void *blob, size_t size) {
  static const uint8_t magic[8] = {'Q','B','P','T','O','K','1',0};
  const uint8_t *p = (const uint8_t *)blob;
  if (!out) return -1;
  out->blob = 0;
  if (!p || size < 64) return -1;
  for (unsigned i = 0; i < 8; ++i) if (p[i] != magic[i]) return -1;
  if (u32(p+8) != 1 || u32(p+12) != size || u32(p+56) || u32(p+60)) return -1;
  uint32_t slots=u32(p+16), vocab=u32(p+20), merges=u32(p+24), added=u32(p+28);
  uint32_t records=u32(p+32), pieces=u32(p+36), piece_size=u32(p+40);
  uint32_t merge_offset=u32(p+44), added_offset=u32(p+48), bytes_offset=u32(p+52);
  if (!slots || vocab > slots || added > slots-vocab || records != 64 ||
      !range(records, (size_t)slots*12, size) ||
      pieces != records+(size_t)slots*12 || !range(pieces,piece_size,size) ||
      merge_offset != ((pieces+(size_t)piece_size+3)&~(size_t)3) ||
      !range(merge_offset,(size_t)merges*16,size) ||
      added_offset != merge_offset+(size_t)merges*16 ||
      !range(added_offset,(size_t)added*4,size) ||
      bytes_offset != added_offset+(size_t)added*4 ||
      !range(bytes_offset,1024,size) || bytes_offset+(size_t)1024 != size) return -1;
  /* Validate records once. Later decoding needs just O(1) ID lookup. */
  uint32_t actual_vocab=0, actual_added=0;
  for (uint32_t i=0; i<slots; ++i) {
    const uint8_t *r=p+records+(size_t)i*12;
    uint32_t flags=u32(r+8);
    if (flags & ~7u || flags == 3 || flags == 4 || flags == 5 || flags == 7 ||
        !range(u32(r),u32(r+4),piece_size) || (flags && !u32(r+4))) return -1;
    actual_vocab += (flags == 1);
    actual_added += ((flags & 2u) != 0);
  }
  if (actual_vocab != vocab || actual_added != added) return -1;
  /* These are executable lookup tables, not just decoder metadata. Reject
   * invalid IDs/order up front instead of returning corrupt token IDs later. */
  uint32_t last_left=0, last_right=0;
  for (uint32_t i=0; i<merges; ++i) {
    const uint8_t *row=p+merge_offset+(size_t)i*16;
    uint32_t left=u32(row), right=u32(row+4), result=u32(row+8);
    if (left>=slots || right>=slots || result>=slots || u32(row+12)>=merges ||
        u32(p+records+(size_t)left*12+8)!=1 ||
        u32(p+records+(size_t)right*12+8)!=1 ||
        u32(p+records+(size_t)result*12+8)!=1 ||
        (i && (left<last_left || (left==last_left && right<=last_right)))) return -1;
    last_left=left; last_right=right;
  }
  uint32_t last_length=0xffffffffu;
  for (uint32_t i=0; i<added; ++i) {
    uint32_t token=u32(p+added_offset+(size_t)i*4);
    if (token>=slots) return -1;
    const uint8_t *row=p+records+(size_t)token*12;
    if (!(u32(row+8)&2u) || u32(row+4)>last_length) return -1;
    last_length=u32(row+4);
  }
  for (uint32_t i=0; i<256; ++i) {
    uint32_t token=u32(p+bytes_offset+(size_t)i*4);
    if (token>=slots) return -1;
    const uint8_t *row=p+records+(size_t)token*12;
    if (u32(row+8)!=1 || u32(row+4)!=1 || p[pieces+u32(row)]!=i) return -1;
  }
  out->blob=p; out->size=size; out->slots=slots;
  out->records_offset=records; out->pieces_offset=pieces; out->pieces_size=piece_size;
  out->merge_offset=merge_offset; out->merge_count=merges;
  out->added_offset=added_offset; out->added_count=added;
  return 0;
}
int qwen_token_piece(const QwenTokenizerResource *r, uint32_t token,
                     const uint8_t **bytes, size_t *count, int *special) {
  if (!r || !r->blob || token>=r->slots || !bytes || !count || !special) return -1;
  const uint8_t *p=r->blob+r->records_offset+(size_t)token*12;
  uint32_t flags=u32(p+8), offset=u32(p), length=u32(p+4);
  if (!(flags & 3u) || !range(offset,length,r->pieces_size)) return -1;
  *bytes=r->blob+r->pieces_offset+offset; *count=length; *special=(flags&4u)!=0;
  return 0;
}
static void replacement(QwenEmit emit, void *context) {
  static const uint8_t bytes[3]={0xef,0xbf,0xbd};
  emit(context,bytes,3);
}
void qwen_decode_bytes(QwenUtf8Decoder *s, const uint8_t *bytes, size_t size,
                       QwenEmit emit, void *context) {
  for (size_t i=0; i<size; ++i) {
    uint8_t b=bytes[i];
    if (s->count) {
      int valid=b>=0x80 && b<=0xbf;
      if (s->count==1) {
        uint8_t first=s->pending[0];
        if ((first==0xe0 && b<0xa0) || (first==0xed && b>0x9f) ||
            (first==0xf0 && b<0x90) || (first==0xf4 && b>0x8f)) valid=0;
      }
      if (valid) {
        s->pending[s->count++]=b;
        if (s->count==s->expected) {
          emit(context,s->pending,s->count); s->count=s->expected=0;
        }
        continue;
      }
      replacement(emit,context); s->count=s->expected=0;
      /* The current byte starts a new sequence after the invalid prefix. */
    }
    if (b<0x80) emit(context,&b,1);
    else if (b>=0xc2 && b<=0xf4) {
      s->pending[0]=b; s->count=1;
      s->expected=b<0xe0 ? 2 : b<0xf0 ? 3 : 4;
    } else replacement(emit,context);
  }
}
int qwen_decode_token(const QwenTokenizerResource *r, QwenUtf8Decoder *s,
                      uint32_t token, int skip_special, QwenEmit emit, void *context) {
  const uint8_t *bytes; size_t count; int special;
  if (!s || !emit || qwen_token_piece(r,token,&bytes,&count,&special)) return -1;
  if (!skip_special || !special) qwen_decode_bytes(s,bytes,count,emit,context);
  return 0;
}
void qwen_decode_finish(QwenUtf8Decoder *s, QwenEmit emit, void *context) {
  if (s->count) replacement(emit,context);
  s->count=s->expected=0;
}
static int append(uint8_t *out, size_t capacity, size_t *used,
                   const uint8_t *bytes, size_t size) {
  if (!range(*used,size,capacity) || (size && !bytes)) return -1;
  for (size_t i=0; i<size; ++i) out[(*used)++]=bytes[i];
  return 0;
}
int qwen_chat_single_turn(uint8_t *out, size_t capacity, size_t *written,
                          const uint8_t *system, size_t system_size, int has_system,
                          const uint8_t *user, size_t user_size, int enable_thinking) {
  size_t used=0;
  if (!out || !written) return -1;
  *written=0;
#define TEXT(s) do { if(append(out,capacity,&used,(const uint8_t *)(s),sizeof(s)-1)) return -1; } while(0)
  if (has_system) {
    TEXT("<|im_start|>system\n");
    if (append(out,capacity,&used,system,system_size)) return -1;
    TEXT("<|im_end|>\n");
  }
  TEXT("<|im_start|>user\n");
  if (append(out,capacity,&used,user,user_size)) return -1;
  TEXT("<|im_end|>\n<|im_start|>assistant\n");
  if (!enable_thinking) { TEXT("<think>\n\n</think>\n\n"); }
#undef TEXT
  *written=used;
  return 0;
}
