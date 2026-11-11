"""Record the required full unittest run, preserving the actual exit code."""
import subprocess,sys
from datetime import datetime,timezone
from collect_v2 import NEWS
p=subprocess.run([sys.executable,'-m','unittest'],capture_output=True,text=True,encoding='utf-8',errors='replace')
text='command: py -m unittest\nstarted/recorded UTC: '+datetime.now(timezone.utc).isoformat()+'\nexit_code: '+str(p.returncode)+'\n'+p.stdout+'\n'+p.stderr
(NEWS/'v2/results/unittest_full_B.txt').write_text(text,encoding='utf-8')
print(text[-9000:]);sys.exit(p.returncode)
