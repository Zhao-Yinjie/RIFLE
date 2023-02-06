import streamlit as st
import numpy as np
from pandas import DataFrame
from keybert import KeyBERT

import seaborn as sns
# For download buttons
from functionforDownloadButtons import download_button
import os
import json

import torch
import os
from torch import nn 
import numpy as np 
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 
from tqdm.notebook import tqdm

# from tqdm import tqdm

class DictObj(object):
    # 私有变量是map
    # 设置变量的时候 初始化设置map
    def __init__(self, mp):
        self.map = mp
        # print(mp)

# set 可以省略 如果直接初始化设置
    def __setattr__(self, name, value):
        if name == 'map':# 初始化的设置 走默认的方法
            # print("init set attr", name ,"value:", value)
            object.__setattr__(self, name, value)
            return
        # print('set attr called ', name, value)
        self.map[name] = value
# 之所以自己新建一个类就是为了能够实现直接调用名字的功能。
    def __getattr__(self, name):
        # print('get attr called ', name)
        return  self.map[name]


Config = DictObj({
    'poem_path' : "./tang.npz",
    'tensorboard_path':'./tensorboard',
    'model_save_path':'./modelDict/poem.pth',
    'embedding_dim':100,
    'hidden_dim':1024,
    'lr':0.001,
    'LSTM_layers':3
})



class PoemDataSet(Dataset):
    def __init__(self,poem_path,seq_len):
        self.seq_len = seq_len
        self.poem_path = poem_path
        self.poem_data, self.ix2word, self.word2ix = self.get_raw_data()
        self.no_space_data = self.filter_space()
        
    def __getitem__(self, idx:int):
        txt = self.no_space_data[idx*self.seq_len : (idx+1)*self.seq_len]
        label = self.no_space_data[idx*self.seq_len + 1 : (idx+1)*self.seq_len + 1] # 将窗口向后移动一个字符就是标签
        txt = torch.from_numpy(np.array(txt)).long()
        label = torch.from_numpy(np.array(label)).long()
        return txt,label
    
    def __len__(self):
        return int(len(self.no_space_data) / self.seq_len)
    
    def filter_space(self): # 将空格的数据给过滤掉，并将原始数据平整到一维
        t_data = torch.from_numpy(self.poem_data).view(-1)
        flat_data = t_data.numpy()
        no_space_data = []
        for i in flat_data:
            if (i != 8292 ):
                no_space_data.append(i)
        return no_space_data
    def get_raw_data(self):
#         datas = np.load(self.poem_path,allow_pickle=True)  #numpy 1.16.2  以上引入了allow_pickle
        datas = np.load(self.poem_path, allow_pickle=True)
        data = datas['data']
        ix2word = datas['ix2word'].item()
        word2ix = datas['word2ix'].item()
        return data, ix2word, word2ix

def check(result):
        
    indexes = [index for index, char in enumerate(result)
               if char == '，']
               
    for i in indexes:
        if ((int(i)-7)%16 != 0 and (int(i)-5)%12 != 0):
            return False
    
    indexes = [index for index, char in enumerate(result)
               if char == '。']
    
    for i in indexes:
        if ((int(i)-15)%16 != 0 and (int(i)-11)%12 != 0):
            return False
            
    return True

poem_ds = PoemDataSet(Config.poem_path, 48)
ix2word = poem_ds.ix2word
word2ix = poem_ds.word2ix

poem_loader =  DataLoader(poem_ds,
                     batch_size=16,
                     shuffle=True,
                     num_workers=0)

# import torch.nn.functional as F
class MyPoetryModel_tanh(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MyPoetryModel_tanh, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)#vocab_size:就是ix2word这个字典的长度。
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.LSTM_layers,
                            batch_first=True,dropout=0, bidirectional=False)
        self.fc1 = nn.Linear(self.hidden_dim,2048)
        self.fc2 = nn.Linear(2048,4096)
        self.fc3 = nn.Linear(4096,vocab_size)
#         self.linear = nn.Linear(self.hidden_dim, vocab_size)# 输出的大小是词表的维度，

    def forward(self, input, hidden=None):
        embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(Config.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(Config.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))#hidden 是h,和c 这两个隐状态
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output,hidden


model = MyPoetryModel_tanh(8293,
                  embedding_dim=Config.embedding_dim,
                  hidden_dim=Config.hidden_dim)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 3
optimizer = optim.Adam(model.parameters(), lr=Config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10,gamma=0.1)#学习率调整
criterion = nn.CrossEntropyLoss()


#因为使用tensorboard画图会产生很多日志文件，这里进行清空操作
import shutil  
if os.path.exists(Config.tensorboard_path):
    shutil.rmtree(Config.tensorboard_path)  
    os.mkdir(Config.tensorboard_path)




#模型保存
# if os.path.exists(Config.model_save_path) == False: 
#     os.mkdir(Config.model_save_path)   
# torch.save(model.state_dict(), Config.model_save_path)

model.load_state_dict(torch.load("model_4.pth"))  # 模型加载

def generate(model, start_words, ix2word, word2ix,device):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    
    #最开始的隐状态初始为0矩阵
    hidden = torch.zeros((2, Config.LSTM_layers*1,1,Config.hidden_dim),dtype=torch.float)
    input = input.to(device)
    hidden = hidden.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
            for i in range(48):#诗的长度
                output, hidden = model(input, hidden)
                # 如果在给定的句首中，input为句首中的下一个字
                if i < start_words_len:
                    w = results[i]
                    input = input.data.new([word2ix[w]]).view(1, 1)
               # 否则将output作为下一个input进行
                else:
                    top_index = output.data[0].topk(1)[1][0].item()#输出的预测的字
                    w = ix2word[top_index]
                    results.append(w)
                    input = input.data.new([top_index]).view(1, 1)
                if w == '<EOP>': # 输出了结束标志就退出
                    del results[-1]
                    break
    return results







st.set_page_config(
    page_title="AI-POEM Generator",
    page_icon="📕",
    layout="centered",
)


def _max_width_():
    max_width_str = f"max-width: 2400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("📕AI-POEM Generator")
    st.header("")



with st.expander("About this app", expanded=True):

    st.write(
        """     
-   The *AI-POEM Generator* app is an demo interface to better demonstrate the results by giving a UI!
-   It uses Machine Learning technique that leverages multiple NLP embeddings to create poems.
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## **📌 Try it yourself **")
with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
            "Choose your model",
            ["Model1 (Default)", "Model2"],
            help="At present, you can choose between 2 models to generate the poems. More to come!",
        )

        if ModelType == "Model1 (Default)":
            # use model 1
            a=1

        else:
            # use another model
            a=2
        



    with c2:
        doc = st.text_area(
            "Enter the start of the poem",
            height=110,
        )

        MAX_WORDS = 10
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 10 words will be reviewed."
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="✨ Generate Poems!")


    StopWords = None

if not submit_button:
    # st.stop()
    a2ewde = 1




userinput = doc

i=0
st.markdown("## **🎈 Check Results **")

st.header("")


model.load_state_dict(torch.load("model_4.pth"))
results = generate(model,userinput, ix2word,word2ix,device)
if(len(results)%4==0 and check(results)):
    st.write("Generated Poem: ")
    st.write(' '.join(i for i in results))
    i+=1


model.load_state_dict(torch.load("model_6.pth"))
results = generate(model,userinput, ix2word,word2ix,device)
if(len(results)%4==0 and check(results)):
    st.write("Generated Poem: ")
    st.write(' '.join(i for i in results))
    if (results[-1] != "。"):
        st.write(''.join('。'))
    i+=1


model.load_state_dict(torch.load("model_9.pth"))
results = generate(model,userinput, ix2word,word2ix,device)
if(len(results)%4==0 and check(results)):
    st.write("Generated Poem: ")
    st.write(' '.join(i for i in results))
    i+=1

if(i==0):
    st.write("Model 9 result: ")
    st.write(' '.join(i for i in results))





cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

