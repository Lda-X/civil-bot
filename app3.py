import streamlit as st
import os
import re
import fitz
import time
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings               
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from zhipuai import ZhipuAI

#配置与初始化
load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

st.set_page_config(
    page_title="民法典智能专家助手",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

VECTOR_STORE_PATH = "./faiss_index_final"
DATA_DIR = "./data"
HISTORY_FILE = "chat_history.json"

#定义CSS
st.markdown("""
<style>
    /* 全局字体优化 */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #F5F7FA !important;
    }
    
    /* 侧边栏标题样式 */
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #303133;
        margin-bottom: 20px;
    }
    
    /* 聊天记录按钮样式 */
    .history-btn {
        text-align: left;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 5px;
        cursor: pointer;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
    }

    /* 推荐问题卡片的样式 */
    div.stButton > button {
        width: 100%;
        height: auto;
        padding: 15px;
        background-color: #F5F7FA;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        color: #303133; 
        text-align: left;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
    }

    div.stButton > button:hover {
        background-color: #e3f2fd;
        border-color: #90caf9;
        color: #1976d2;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* 聊天气泡优化 */
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
    }

    /* 隐藏 Streamlit 默认的 deploy 按钮 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* st.info的样式 */
    .stAlert {
        width: 100% !important; 
        background-color: #FFFFFF !important; 
        border: 1px solid #FFFFFF !important; 
        border-radius: 8px; 
        transition: all 0.3s ease; 
    }
    /* 悬浮效果 */
    .stAlert:hover {
        transform: translateY(-3px); 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); 
        background-color: #FFFFFF !important; 
    }
    .stAlert > div {
        text-align: center !important;
        color: #2c3e50 !important;
    }
    /* 参数调节卡片样式 */
    .param-card {
        background: linear-gradient(135deg, #6A7FC0 0%, #3a4a9a 100%);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }

    .param-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }

    .param-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }

    .param-title {
        font-size: 14px;
        font-weight: 600;
        color: white;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .param-value {
        font-size: 12px;
        color: rgba(255,255,255,0.9);
        background: rgba(255,255,255,0.1);
        padding: 2px 8px;
        border-radius: 10px;
        min-width: 40px;
        text-align: center;
    }

    .param-icon {
        font-size: 16px;
    }
    /* 滑块自定义样式 */
    .stSlider > div > div > div > div {
        background: linear-gradient(to right, #2196f3, #64b5f6) !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<script>
// 实时更新参数值显示
function updateParamValues() {
    // 获取滑块值并更新显示
    const tempSlider = document.querySelector('input[aria-label="Temperature"]');
    const topPSlider = document.querySelector('input[aria-label="Top_P"]');

    if (tempSlider) {
        const tempValue = document.getElementById('temp-value');
        if (tempValue) tempValue.textContent = tempSlider.value;
    }

    if (topPSlider) {
        const topPValue = document.getElementById('top-p-value');
        if (topPValue) topPValue.textContent = topPSlider.value;
    }

}

// 监听滑块变化
document.addEventListener('input', function(e) {
    if (e.target.type === 'range') {
        updateParamValues();
    }
});

// 页面加载时初始化
window.addEventListener('load', updateParamValues);
</script>
""", unsafe_allow_html=True)

#对话管理
def load_history_from_disk():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_history_to_disk():
    if "all_chats" in st.session_state:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.all_chats, f, ensure_ascii=False, indent=2)

def create_new_chat():
    new_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.all_chats[new_id] = {
        "title": f"新对话 {timestamp}",
        "messages": [],
        "created_at": timestamp
    }
    st.session_state.current_chat_id = new_id
    save_history_to_disk()
    return new_id

def delete_chat(chat_id):
    if chat_id in st.session_state.all_chats:
        del st.session_state.all_chats[chat_id]
        if st.session_state.current_chat_id == chat_id:
            st.session_state.current_chat_id = None
        save_history_to_disk()
# 初始化 Session State
if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_history_from_disk()

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.all_chats:
    if len(st.session_state.all_chats) > 0:
        st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[-1]
    else:
        create_new_chat()
#数据解析逻辑
#解析《民法典》
def parse_civil_code_articles(pdf_path):
    docs = []
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        pattern = r"(第[零一二三四五六七八九十百千]+条\s)"
        segments = re.split(pattern, full_text)

        for i in range(1, len(segments), 2):
            if i + 1 < len(segments):
                article_title = segments[i].strip()
                content = segments[i + 1].strip()
                full_text = f"{article_title}：{content}"

                docs.append(Document(
                    page_content=full_text,
                    metadata={"type": "article", "source": "民法典", "article": article_title}
                ))
        doc.close()
    except Exception as e:
        st.error(f"解析民法典失败: {e}")
    return docs

#解析配套读物
def parse_study_books(data_dir):
    all_docs = []

    if not os.path.exists(data_dir):
        return []

    files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "；", " "]
    )

    for file in files:
        if "民法典.pdf" in file or file == "民法典.pdf":
            continue

        file_path = os.path.join(data_dir, file)
        try:
            doc = fitz.open(file_path)
            file_text = ""
            for page in doc:
                file_text += page.get_text() + "\n"
            doc.close()

            if not file_text.strip():
                st.warning(f"文件{file}是扫描件，无法解析")
                continue

            chunks = text_splitter.split_text(file_text)

            for chunk in chunks:
                doc_type = "explanation"
                if "案例" in chunk or "判决" in chunk:
                    doc_type = "case"
                elif "风险" in chunk or "提示" in chunk:
                    doc_type = "risk_tip"

                all_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "type": doc_type,
                        "source": file,
                        "is_book": True
                    }
                ))

        except Exception as e:
            st.warning(f"解析书本 {file} 出错: {e}")

    return all_docs

#向量库构建
def build_vector_store_safe(docs):
    embeddings = ZhipuAIEmbeddings(model="embedding-2", api_key=ZHIPU_API_KEY)

    progress_bar = st.progress(0)
    status_text = st.empty()

    batch_size = 20
    total_docs = len(docs)
    vector_store = None

    st.info(f"开始向量化，共 {total_docs} 个片段，将分批处理...")

    for i in range(0, total_docs, batch_size):
        batch = docs[i: i + batch_size]

        try:
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)

            current_progress = min((i + batch_size) / total_docs, 1.0)
            progress_bar.progress(current_progress)
            status_text.text(f"正在处理: {i + len(batch)} / {total_docs}")

            time.sleep(0.1)

        except Exception as e:
            st.error(f"在处理第 {i} 到 {i + batch_size} 条数据时出错: {e}")
            if vector_store:
                vector_store.save_local(VECTOR_STORE_PATH)
                st.warning("已紧急保存当前进度")
            return None

    if vector_store:
        vector_store.save_local(VECTOR_STORE_PATH)
        status_text.text("处理完成！")
        return vector_store
    return None
#对话与Prompt逻辑
def get_zhipu_chat_response(prompt, temperature=0.5, top_p=0.9,do_stream=True):
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    response = client.chat.completions.create(
        model="glm-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        stream=do_stream
    )
    return response

def check_is_legal_query(query):
    if len(query) < 4 and query in ["你好", "在吗", "hi", "hello", "您好"]:
        return False
        
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    prompt = f"""
    请判断用户的输入是否与【法律咨询、民法典、司法案例、维权】相关。
    用户输入："{query}"
    
    只需要回答：是 或 否
    """
    try:
        response = client.chat.completions.create(
            model="glm-3-turbo", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=5
        )
        result = response.choices[0].message.content.strip()
        return "是" in result
    except:
        return True
#界面逻辑
#初始化Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    if os.path.exists(VECTOR_STORE_PATH):
        try:
            emb = ZhipuAIEmbeddings(model="embedding-2", api_key=ZHIPU_API_KEY)
            st.session_state.vector_store = FAISS.load_local(VECTOR_STORE_PATH, emb,allow_dangerous_deserialization=True)
        except:
            pass

#左侧边栏 (1/4)
with st.sidebar:
    st.markdown('<div class="sidebar-title">对话管理</div>', unsafe_allow_html=True)

    #新建对话按钮
    if st.button("➕开始新对话", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()

    #历史对话列表
    chat_ids = list(st.session_state.all_chats.keys())
    chat_ids.reverse()
    chat_titles = {cid: st.session_state.all_chats[cid]["title"] for cid in chat_ids}
    selected_chat_id = st.selectbox(
        "历史记录",
        options=chat_ids,
        format_func=lambda x: chat_titles[x],
        index=chat_ids.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in chat_ids else 0,
        key="history_select"
    )

    #切换对话逻辑
    if selected_chat_id != st.session_state.current_chat_id:
        st.session_state.current_chat_id = selected_chat_id
        st.rerun()

    #删除当前对话
    col_del1, col_del2 = st.columns(2)
    with col_del1:
        if st.button("删除本条", use_container_width=True):
            delete_chat(st.session_state.current_chat_id)
            if not st.session_state.all_chats:
                create_new_chat()
            else:
                st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[-1]
            st.rerun()
    with col_del2:
        if st.button("清空所有", use_container_width=True):
            st.session_state.all_chats = {}
            create_new_chat()
            st.rerun()

    st.markdown("---")
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">大模型参数调节</div>',
        unsafe_allow_html=True
    )

    # 参数控制
    st.markdown(
        '''
        <div class="param-card">
            <div class="param-header">
                <div class="param-title">
                    <span class="param-icon">🌡️</span>
                    <span>随机性（Temperature）</span>
                </div>
                <div class="param-value" id="temp-value">0.5</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    col_temp1, col_temp2 = st.columns([3, 1])
    with col_temp1:
        temperature = st.slider(
            "随机性（Temperature）",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            label_visibility="collapsed",
            help="值越高回答越多样，值越低回答越确定。",
            key="temp_slider"
        )
    with col_temp2:
        st.markdown(
            f'<div style="text-align: center; padding-top: 8px; color: #666; font-weight: 500;">{temperature}</div>',
            unsafe_allow_html=True)

    # Top-P
    st.markdown(
        '''
        <div class="param-card">
            <div class="param-header">
                <div class="param-title">
                    <span class="param-icon">🎯</span>
                    <span>多样性（Top-P）</span>
                </div>
                <div class="param-value" id="top-p-value">0.9</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    col_topp1, col_topp2 = st.columns([3, 1])
    with col_topp1:
        top_p = st.slider(
            "多样性（Top-P）",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.1,
            label_visibility="collapsed",
            help="值越低回答越集中，值越高回答越多样。",
            key="top_p_slider"
        )
    with col_topp2:
        st.markdown(f'<div style="text-align: center; padding-top: 8px; color: #666; font-weight: 500;">{top_p}</div>',
                    unsafe_allow_html=True)

    do_stream = st.toggle("流式输出", value=True)

    st.markdown("---")

    st.markdown('<div class="sidebar-title">知识库管理</div>', unsafe_allow_html=True)

    # 知识库状态
    if st.session_state.vector_store:
        st.success("✅ 知识库状态：已挂载")
    else:
        st.warning("⚠️ 知识库状态：未就绪")

    if st.button("🔄 重建或更新知识库", use_container_width=True):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        all_docs = []
        # 解析民法典
        cc_path = os.path.join(DATA_DIR, "民法典.pdf")
        if os.path.exists(cc_path):
            all_docs.extend(parse_civil_code_articles(cc_path))
        # 解析读本
        all_docs.extend(parse_study_books(DATA_DIR))

        if all_docs:
            vs = build_vector_store_safe(all_docs)
            if vs:
                st.session_state.vector_store = vs
                st.toast("知识库构建完成！", icon="🎉")
                time.sleep(1)
                st.rerun()
        else:
            st.error("未找到数据，请检查 data 目录。")

#右侧主区域
#标题区
current_chat = st.session_state.all_chats[st.session_state.current_chat_id]
current_messages = current_chat["messages"]

st.markdown(
    '<div class="main-header" style="text-align: center; font-size: 28px; font-weight: bold;">民法典智能专家助手 <span style="font-size:16px;color:#4a90e2;padding:2px 8px;border-radius:10px;"></span></div>',
    unsafe_allow_html=True
)

#聊天历史展示区
#聊天历史展示区
st.markdown("<br>", unsafe_allow_html=True)
chat_container = st.container()

with chat_container:
    if not current_messages:
        st.info("您好！我是您的民法典小助手。您可以点击下方的快捷卡片，或直接输入问题。")

    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # ✅兼容旧 chat_history.json：旧消息没有 sources 字段也不会报错
            if msg.get("role") == "assistant":
                sources = msg.get("sources", None)

                # ✅为了“每条回答都有参考来源区域”，即使 sources 为空也显示
                with st.expander("参考来源", expanded=False):
                    if sources and isinstance(sources, list) and len(sources) > 0:
                        st.write("本次回答参考了以下文档：")
                        for s in sources:
                            st.caption(f"• {s}")
                    else:
                        st.caption("（本次未检索到可展示的来源，或该回答未记录来源）")


st.markdown("---")

#推荐问题区
final_prompt = None

st.markdown("**试试这样问**")
suggestions =[
    "邻居装修把我家墙震裂了，怎么索赔？",
    "离婚时，婚前买的房子怎么分？",
    "微信聊天记录能当做借钱的证据吗？",
    "小区高空坠物砸坏车，找不到人谁负责？"
]
col1, col2 = st.columns(2)
selected_suggestion = None

with col1:
    if st.button(suggestions[0], use_container_width=True):
        selected_suggestion = suggestions[0]
    if st.button(suggestions[2], use_container_width=True):
        selected_suggestion = suggestions[2]

with col2:
    if st.button(suggestions[1], use_container_width=True):
        selected_suggestion = suggestions[1]
    if st.button(suggestions[3], use_container_width=True):
        selected_suggestion = suggestions[3]

if selected_suggestion:
    final_prompt = selected_suggestion
#输入区
user_input = st.chat_input("请输入您的问题，支持Enter发送")
if user_input:
    final_prompt = user_input

if final_prompt:
    if not st.session_state.vector_store:
        st.toast("请先在左侧构建知识库！", icon="⚠️")
    else:
        current_messages.append({"role": "user", "content": final_prompt})
        if len(current_messages) == 1:
            new_title = final_prompt[:10] + "..." if len(final_prompt) > 10 else final_prompt
            st.session_state.all_chats[st.session_state.current_chat_id]["title"] = new_title

        save_history_to_disk()
        st.rerun()

if current_messages and current_messages[-1]["role"] == "user":
    last_user_msg = current_messages[-1]["content"]

    with chat_container:
        is_legal = check_is_legal_query(last_user_msg)
        list_articles = []
        list_explanation = []
        list_case = []
        list_risk = []
        ref_sources = set()
        if is_legal:
            with st.spinner("正在查阅民法典..."):
                docs = st.session_state.vector_store.similarity_search(last_user_msg, k=3)

                # 2. 遍历文档进行分类
                for d in docs:
                    src = d.metadata.get('source', '未知来源')
                    type_ = d.metadata.get('type', '未知')
                    article_title = d.metadata.get('article', '')
                    if article_title:
                        ref_sources.add(f"{src} - {article_title}")
                    else:
                        ref_sources.add(f"{src} ({type_})")
                        
                    content = d.page_content
                    if type_ == "article":
                        list_articles.append(content)
                    elif type_ == "case":
                        list_case.append(content)
                    elif type_ == "risk_tip":
                        list_risk.append(content)
                    else:
                        list_explanation.append(content)
        else:
            pass

        # 3. 准备 Prompt 所需的变量
        context_articles = list_articles if list_articles else ["暂无直接相关法律条文"]
        context_explanation = "\n".join(list_explanation) if list_explanation else "暂无详细解读"
        context_case = "\n".join(list_case) if list_case else "暂无相关案例"
        context_risk_tip = "\n".join(list_risk) if list_risk else "暂无风险提示"
    
        context_application_point = ""
        context_main_point = ""
        context_scenario = ""
        history_str = ""
        recent_history = current_messages[:-1][-4:] 
        
        if recent_history:
            history_str = "\n**【历史对话参考】：**\n"
            for msg in recent_history:
                role_label = "用户" if msg["role"] == "user" else "AI助手"
                clean_content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                history_str += f"{role_label}：{clean_content}\n"
        else:
            history_str = "（无历史对话）"
        prompt = last_user_msg
    
        #Prompt
        system_prompt = f"""
        你是一位经验丰富的中国法律专家，精通《中华人民共和国民法典》及其配套的权威解读、司法案例、生活场景示例和风险提示。
        ### 🛑 核心指令（请务必优先执行）：
        请先判断用户的【输入意图】：
        👉 **情况一：如果是日常问候、闲聊或无具体语义的输入**（例如："你好"、"在吗"、"你是谁"、"Hi"）：
            - 请直接用亲切、自然的语气回复。
            - 简要介绍你的身份（民法典智能助手），并引导用户提问法律问题。
            - **严禁**使用下方的法律回答模板，**忽略**上下文信息。
        👉 **情况二：如果是法律咨询、具体问题或搜索请求**：
            - 请结合【上下文信息】，**严格**按照以下结构进行专业解答：
        -------------------------------------------------
        【法律咨询回答结构】
            ### 1. 👨‍⚖️ **权威法律分析**
            - **法律条文依据**：优先引用《民法典》原文。请明确指出是“第XXX条”。
            - **立法原意与司法解释**：结合检索到的专家解读，阐述该条文的立法精神和司法实践中的理解。
            - **核心要点**：提炼条文主旨和关键的适用要点。
    
            ### 2. 💡 **情景化解读与案例说明**
            - **生活化场景模拟**：将抽象的法律条文，通过一个贴近用户生活或工作场景的**具体示例**来阐述。
            - **典型案例分析**：引用检索到的真实案例，说明法律在实践中的具体应用方式、责任划分及法律后果。
            - **风险规避**：根据检索到的风险提示，告知用户在类似情境下可能存在的风险点。
    
            ### 3. ✅ **专业行动建议**
            - 基于以上分析，提供1-3条可操作的、具有建设性的行动建议。
    
        ---
        **【可参考的上下文信息】**
    
        **《民法典》原文片段：**
        {chr(10).join(context_articles)}
    
        **专家解读与适用要点：**
        {context_explanation}
        {context_application_point}
        {context_main_point}
        {history_str}
        **典型案例与生活场景：**
        {context_case}
        {context_scenario}
            
        **相关风险提示：**
        {context_risk_tip}
            
        **【用户问题】：**
        {prompt}
    """
    
    # 生成回答
    with st.chat_message("assistant", avatar="⚖️"):
        placeholder = st.empty()
        full_response = ""
    
        try:
            stream = get_zhipu_chat_response(system_prompt, temperature, top_p, do_stream)
    
            if do_stream:
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            else:
                full_response = stream.choices[0].message.content
                placeholder.markdown(full_response)
                
            if ref_sources:
                with st.expander("参考来源"):
                    st.write("本次回答参考了以下文档：")
                    for src in ref_sources:
                        st.caption(f"• {src}")
    
            st.session_state.all_chats[st.session_state.current_chat_id]["messages"].append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "sources": sorted(list(ref_sources)) if ref_sources else []
                }
            )
            save_history_to_disk()
    
    
        except Exception as e:
    
            st.error(f"生成回答出错: {e}")








