import os
import sys
import re

data = sys.stdin.read()
lines = data.split('\n')

pkgPat = " [a-zA-Z0-9\-]* : Depends: .*"
pkgNamePat = " [a-zA-Z0-9\-]* :"

relLines = []
for line in lines:
    if(re.match(pkgPat, line) is not None):
        relLines.append(line)

for line in relLines:
    pkg = re.search(pkgNamePat, line)[0][1:-2]
    if pkg is not None:
        os.system('sudo dpkg -r --force-all ' + pkg)
    else:
        print("None: " + line)