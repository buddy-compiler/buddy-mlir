import pathlib,json,hashlib,os,fcntl
root=pathlib.Path('/home/hjuser/Desktop/fpga-tester-ISCAS')
assert root.resolve()==root and not (root/'fpga-runs').is_symlink()
lock=os.open(str(root/'.fpga_run.lock'),os.O_RDWR|os.O_NOFOLLOW)
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
expected={'run-015b4fabf4014edc': [{'file': 'image.bin', 'size': 5849856, 'sha256': '8e2747d650a4e9e10efbfd212230b26c8e00aad71614d3c6bd0a1df5b06df517'}, {'file': 'weights-w8a8.bin', 'size': 171981568, 'sha256': '7dadd6dc021fb23eb9741f6d2da3dc00c31b9b1e5c7b5369b2b705d97de98143'}], 'run-9173a2481e114612': [{'file': 'image.bin', 'size': 5860032, 'sha256': 'b8ffaef85a845a759d1477e5284132adea2e8da1942d5620829929d09b7d1030'}, {'file': 'weights-w8a8.bin', 'size': 171981568, 'sha256': '7dadd6dc021fb23eb9741f6d2da3dc00c31b9b1e5c7b5369b2b705d97de98143'}], 'run-390dfc362f234ec6': [{'file': 'image.bin', 'size': 5858048, 'sha256': '6fde5a81a5abfd24bd7e69dce767c1df909d156b009d07c2710fc53d1baed6bd'}, {'file': 'weights-w8a8.bin', 'size': 171981568, 'sha256': '7dadd6dc021fb23eb9741f6d2da3dc00c31b9b1e5c7b5369b2b705d97de98143'}], 'run-53885c89b85646dc': [{'file': 'image.bin', 'size': 6694208, 'sha256': '26d089f14274de36236b0fd1c573cedc2ce1f230e88527d3d4abd0364823eca2'}, {'file': 'weights-w8a8.bin', 'size': 219342592, 'sha256': '4ae30d1197e0531834dd8d81f4bf92c78f435e0267eee165cc4910c0a27ef4c5'}, {'file': 'tokenizer.bin', 'size': 5222976, 'sha256': '7e436961bdc35adbcca4887a6f767cdbdaabc69cc0349298a40c54a400b4bcc4'}], 'run-06aaa50105af47e1': [{'file': 'image.bin', 'size': 6703168, 'sha256': '74cc8ce95de55c3e897800be609d5d94a39c5af58c8d8cb2a1bc0b65f6377c5e'}, {'file': 'weights-w8a8.bin', 'size': 219342592, 'sha256': '4ae30d1197e0531834dd8d81f4bf92c78f435e0267eee165cc4910c0a27ef4c5'}, {'file': 'tokenizer.bin', 'size': 5222976, 'sha256': '7e436961bdc35adbcca4887a6f767cdbdaabc69cc0349298a40c54a400b4bcc4'}]}
out={'before_available_bytes':os.statvfs(str(root)).f_bavail*os.statvfs(str(root)).f_frsize,'files':[]}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
paths=[]
for run,segments in expected.items():
 directory=root/'fpga-runs'/run
 assert not directory.is_symlink() and directory.resolve().parent==(root/'fpga-runs')
 assert json.loads((directory/'result.json').read_text())['status'] in ('OK','ERROR','INTERRUPTED')
 for segment in segments:
  assert segment['file'] in ('image.bin','weights-w8a8.bin','tokenizer.bin')
  source=directory/segment['file']
  assert not source.is_symlink() and sha(source)==segment['sha256']
  for p in (source,directory/(segment['file']+'.readback')):
   assert not p.is_symlink() and p.is_file()
   out['files'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'expected_full_sha256':segment['sha256']})
   paths.append(p)
print(json.dumps(out),flush=True)
for p in paths:p.unlink()
print(json.dumps({'removed_bytes':sum(x['bytes'] for x in out['files']),'after_available_bytes':os.statvfs(str(root)).f_bavail*os.statvfs(str(root)).f_frsize}),flush=True)
