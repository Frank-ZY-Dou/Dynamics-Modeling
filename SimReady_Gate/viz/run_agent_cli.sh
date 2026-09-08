#!/bin/bash
# the agent loop through the CLI on the heaped layout: summarize -> check -> repair --out -> settle, outputs kept as JSON
cd "$(dirname "$0")/.."
P=${PYTHON:-python3}; D=results/videos/pile_n20; L=$D/layout.json; PR=$D/program.json
$P -m simready.cli summarize $L > $D/agent_summarize.txt 2>/dev/null
$P -m simready.cli check $L $PR > $D/agent_check.txt 2>/dev/null; echo "check exit=$?" >> $D/agent_check.txt
$P -m simready.cli repair $L $PR --out $D/pile_repaired.json --s-min ${S_MIN:-0.05} > $D/agent_repair.txt 2>/dev/null; echo "repair exit=$?" >> $D/agent_repair.txt
$P -m simready.cli settle $D/pile_repaired.json --program $PR > $D/agent_settle.txt 2>/dev/null; echo "settle exit=$?" >> $D/agent_settle.txt
echo "agent cli run done"
