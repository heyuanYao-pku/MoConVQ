import gradio as gr
import tokenize_motion as tokenizer
import decode_token as decoder

from MoConVQCore.Utils.motion_dataset import MotionDataSet


import pandas
import numpy as np
import tempfile


def get_model():
    args = tokenizer.build_args(args_in=['_dummy'])
    agent, env = tokenizer.get_model(args)
    return agent, env

agent, simu_env = get_model()
    
def encode(fn, flip_flag):
    fn = str(fn)
    motion_data = MotionDataSet(20)
    
    motion_data.add_bvh_with_character(fn, simu_env.sim_character, flip=flip_flag)    
    seq_indices = tokenizer.encode(agent, motion_data.observation)
    
    seq_indices = seq_indices.transpose()
    # append label as index of the rows
    seq_indices = np.concatenate((np.arange(seq_indices.shape[0]).reshape(-1, 1), seq_indices), axis=1)
    
    # for visualize
    return pandas.DataFrame(seq_indices, 
                            columns=['index'] + [f'RVQ-{i}' for i in range(seq_indices.shape[1]-1)])
    
    # return seq_indices
    
def save_motion_token(data: pandas.DataFrame):
    fn = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name
    data.iloc[:,1:].to_csv(fn, index=False)
    
    return fn
    

def decode(input_token_file, previous_tokens):
    # print('input_token_file', input_token_file)
    # print('previous_tokens', previous_tokens)
    
    if input_token_file is not None:
        
        if input_token_file.endswith('.csv'):            
            seq_indices = np.asarray(pandas.read_csv(input_token_file)).transpose()
        else:
            # with open(input_token_file, 'r') as f:
            #     indices = [int(x) for x in f.read().split()]
            
            # input_token_level = int(round(input_token_level))
            # if len(indices) % input_token_level > 0:
            #     indices += [0]*(input_token_level - len(indices) % input_token_level)            
            # seq_indices = np.asarray(indices).reshape(input_token_level, -1)
            seq_indices = np.loadtxt(input_token_file)
            
    else:
        seq_indices = np.asarray(previous_tokens.iloc[:,1:]).transpose()
        
    print('seq_indices', seq_indices.shape)
    
    out_fn = tempfile.NamedTemporaryFile(suffix='.bvh', delete=False).name
    
    saver = decoder.decode(agent, seq_indices)    
    saver.to_file(out_fn)
    
    return out_fn
    
with gr.Blocks(title='Motion Tokenizer and Decoder', theme=gr.themes.Soft()) as demo:    
    gr.Markdown('# [MoConVQ] Motion Tokenizer and Decoder')        
    
    with gr.Tab('[+] Tokenize a motion (bvh)'):
        with gr.Row():
            input_fn = gr.File(label='Motion File (bvh)', file_count='single', file_types=['.bvh'])    
        with gr.Row():            
            with gr.Column():
                flip_flag = gr.Checkbox(value=False, label='flip motion')
            with gr.Column():
                encode_btn = gr.Button('Tokenize!')
                    
        with gr.Row():
            motion_tokens = gr.DataFrame(datatype='number', visible=False)
            
        with gr.Row():
            save_token_btn = gr.Button('Save Tokens', visible=False)
        with gr.Row():
            save_token_file = gr.File(label='Motion Token File (csv)', file_count='single', file_types=['.csv'], visible=False) 

    encode_btn.click(encode, [input_fn, flip_flag], motion_tokens)
    encode_btn.click(lambda: gr.DataFrame(datatype='number', visible=True), outputs=motion_tokens)
    encode_btn.click(lambda: gr.Button('Save Tokens', visible=True), outputs=save_token_btn)
    save_token_btn.click(lambda: gr.File(label='Motion Token File (csv)', file_count='single', file_types=['.csv'], visible=True), outputs=save_token_file)
    save_token_btn.click(save_motion_token, [motion_tokens], save_token_file)
    
    with gr.Tab('[+] Decode a motion'):
        with gr.Row():
            with gr.Column():
                upload_token_btn = gr.Button('Upload Motion Tokens')
            with gr.Column():
                use_encoded_token_btn = gr.Button('Use Tokens from Previous Tab')
                
        with gr.Row():
            input_token_file = gr.File(visible=False)            
        # with gr.Row():
        #     input_token_level = gr.Number(visible=False)
        with gr.Row():
            previous_tokens = gr.DataFrame(value=[], visible=False)
                
        with gr.Row():
            decoder_btn = gr.Button('Decode!')
        with gr.Row():
            output_file = gr.File(visible=False)
                        
        upload_token_btn.click(lambda: gr.File(label='Motion Token File (csv)', file_count='single', file_types=['.csv', '.txt'], visible=True), outputs=input_token_file)
        upload_token_btn.click(lambda: gr.DataFrame(visible=False), outputs=previous_tokens)
        # upload_token_btn.click(lambda: gr.Number(1, label='Token RVQ Level', precision=0, minimum=1, maximum=8, step=1, visible=True), outputs=input_token_level)
        
        use_encoded_token_btn.click(lambda x: gr.DataFrame(x, datatype='number', visible=True), inputs=motion_tokens, outputs=previous_tokens)
        use_encoded_token_btn.click(lambda: gr.File(visible=False), outputs=input_token_file)
        
        decoder_btn.click(lambda: gr.File(label='Motion File (bvh)', file_count='single', file_types=['.bvh'], visible=True), outputs=output_file)
        decoder_btn.click(decode, inputs=[input_token_file, previous_tokens], outputs=output_file)
        
    
demo.launch()