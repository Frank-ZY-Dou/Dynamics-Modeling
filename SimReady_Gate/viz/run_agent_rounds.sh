#!/bin/bash
# the agent loop through the CLI on the heaped layout, three recorded rounds:
#   program_round1.json -> agent_round1_*   program_round2.json -> agent_round2_*   program.json -> agent_* (accepted)
cd "$(dirname "$0")/.."
P=${PYTHON:-python3}; D=${1:-results/videos/pile_n20}; L=$D/layout.json; SMIN=${S_MIN:-0.3}
$P -m simready.cli summarize $L > $D/agent_summarize.txt 2>/dev/null
run_round () {   # $1 = program file, $2 = output prefix
  $P -m simready.cli check $L $1 > $D/$2check.txt 2>/dev/null; echo "check exit=$?" >> $D/$2check.txt
  $P -m simready.cli repair $L $1 --out $D/$2repaired.json --s-min $SMIN > $D/$2repair.txt 2>/dev/null; echo "repair exit=$?" >> $D/$2repair.txt
  $P -m simready.cli settle $D/$2repaired.json --program $1 > $D/$2settle.txt 2>/dev/null; echo "settle exit=$?" >> $D/$2settle.txt
}
for f in $D/program_round*.json; do [ -f "$f" ] || continue; k=$(basename $f .json | sed 's/program_round//'); run_round $f agent_round${k}_; done
run_round $D/program.json agent_
cp $D/agent_repaired.certificate.json $D/pile_repaired.certificate.json 2>/dev/null
echo "agent rounds done"
