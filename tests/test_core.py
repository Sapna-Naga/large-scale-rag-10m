import json
import subprocess
import sys
from rag_system.core import document, chunks, metrics, rrf, point_id, records
from rag_system.console import DailyLog
import pytest


def test_document():
    assert document({'_id':'x','text':' hello '}) == ('x','','hello')
    with pytest.raises(ValueError):
        document({'id':'x','text':''})


def test_chunk_offsets():
    def tokenizer(text, **kwargs):
        return {'offset_mapping':[(i,i+1) for i in range(len(text))]}
    assert list(chunks('abcde',tokenizer,3,1)) == [('abc',0,3),('cde',2,5)]


def test_metrics():
    m=metrics(['a','a','b'],{'b':1},2)
    assert m['recall']==1 and m['precision']==.5 and m['mrr']==.5
    assert metrics(['z'],{'a':1},1)['ndcg']==0


def test_rrf():
    a,b,c=[{'point_id':x} for x in 'abc']
    assert rrf([[a,b],[c,b]])[0]['point_id']=='b'
    assert point_id('a',1)==point_id('a',1)!=point_id('a',2)


def test_records(tmp_path):
    p=tmp_path/'input.jsonl'
    p.write_text(json.dumps({'id':'x'})+'\n')
    assert list(records(p))==[{'id':'x'}]
    p.write_text('bad')
    with pytest.raises(ValueError):
        list(records(p))


def test_daily_append(tmp_path):
    log=DailyLog(tmp_path)
    log.write('INPUT','first')
    log.write('OUTPUT','second')
    files=list(tmp_path.glob('*.log'))
    assert len(files)==1
    content=files[0].read_text()
    assert 'first' in content and 'second' in content and '+05:30' in content


def test_console_subprocess(tmp_path):
    cmd=[sys.executable,'-m','rag_system.console','--log-dir',str(tmp_path),'--',sys.executable,'-c','import sys; print(input()); print("error stream",file=sys.stderr); sys.exit(3)']
    result=subprocess.run(cmd,input='hello\n',text=True,capture_output=True,timeout=15)
    assert result.returncode==3
    assert 'hello' in result.stdout and 'error stream' in result.stdout
    log=next(tmp_path.glob('*.log')).read_text()
    assert '[INPUT] hello' in log and '[EXIT] 3' in log
