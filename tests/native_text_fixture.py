"""Local four-token stateful GenAI fixture. No pretrained model or download."""
import json
import numpy as np
import openvino as ov
from openvino import opset13 as ops
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast
import openvino_tokenizers


def create_fixture(path):
    tok = Tokenizer(models.WordLevel({'<unk>':0,'a':1,'b':2,'<eos>':3}, unk_token='<unk>'))
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='<unk>', eos_token='<eos>')
    hf.save_pretrained(str(path))
    tokenizer, detokenizer = openvino_tokenizers.convert_tokenizer(hf, with_detokenizer=True)
    ov.serialize(tokenizer, path/'openvino_tokenizer.xml')
    ov.serialize(detokenizer, path/'openvino_detokenizer.xml')
    ids = ops.parameter([-1,-1], ov.Type.i64, name='input_ids')
    mask = ops.parameter([-1,-1], ov.Type.i64, name='attention_mask')
    positions = ops.parameter([-1,-1], ov.Type.i64, name='position_ids')
    beam = ops.parameter([-1], ov.Type.i32, name='beam_idx')
    for n in (ids,mask,positions,beam): n.output(0).set_names({n.friendly_name})
    value = ops.gather(ops.constant(np.arange(16,dtype=np.float32).reshape(4,4)/20),ids,ops.constant(0))
    sinks = []
    for i in range(4):
        info = ov.op.util.VariableInfo()
        info.data_type = ov.Type.f32
        info.data_shape = ov.PartialShape([-1,1,-1,4])
        info.variable_id = f'past_key_values.{i}.key'
        variable = ov.op.util.Variable(info)
        read = ops.read_value(ops.constant(np.zeros([1,1,0,4],dtype=np.float32)),variable)
        previous = ops.gather(read,beam,ops.constant(0))
        update = ops.concat([previous,ops.unsqueeze(value,ops.constant([1]))],2)
        sinks.append(ops.assign(update,variable))
        cache_sum = ops.reduce_sum(previous, ops.constant([0,1,2,3]), False)
        value = ops.add(ops.matmul(value,ops.constant(np.eye(4,dtype=np.float32)),False,False),
                        ops.multiply(cache_sum,ops.constant(np.float32(.0001))))
    logits = ops.matmul(value,ops.constant(np.array([[0,1,0,0]]*4,dtype=np.float32)),False,False)
    logits.output(0).set_names({'logits'})
    model = ov.Model([ops.result(logits)],sinks,[ids,mask,positions,beam])
    ov.serialize(model,path/'openvino_model.xml')
    (path/'config.json').write_text(json.dumps({'model_type':'llama','num_attention_heads':1,'num_key_value_heads':1,'hidden_size':4,'num_hidden_layers':4,'vocab_size':4}))
    (path/'generation_config.json').write_text(json.dumps({'eos_token_id':3,'pad_token_id':0}))

