import os
import time
import pickle
from dotenv import load_dotenv

from markdown2 import markdown
from markupsafe import Markup
from flask import Flask, request, render_template, jsonify
import subprocess
from typing import Optional

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_transformers import LongContextReorder

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

# from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")
os.makedirs('database', exist_ok=True)

CHAR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 사용자의 질문에서 캐릭터 또는 인물(예: 장원영, 카리나, 윈터, 우기, 츄, 백현 등)과 사용자의 코딩 수준(예: Beginner, Intermediate, Advanced, Expert)을 추출하는 모델입니다. 참고로 질문은 한국어로 주어집니다."),
        ("user", "#Format: {format_instruction}\n\n#Question: {query}"),
    ]
)

PROBLEM_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 사용자가 코딩 문제를 요청하고 있는지 판단하는 모델입니다. "
                   "예를 들어, '문제를 만들어 주세요', '문제를 제공해 주세요'와 같은 요청이면 True, "
                   "'설명해 주세요', '알려 주세요'와 같은 요청이면 False로 판단합니다."),
        ("user", "#형식: {format_instruction}\n\n#질문: {query}")
    ]
)


CHAT_TEMPLATE = """
당신은 사용자가 지정한 캐릭터입니다.  
항상 사용자의 코딩 수준에 맞춰 대답하세요.  

1. 대화를 친근하게 만들기 위해 캐주얼한 언어를 사용하세요.  
2. 사용자의 수준에 맞춰 Python 관련 질문에 답변하세요.  
3. 개념을 쉽게 설명하고 예제를 제공하며, 캐릭터의 특징(일상, 말버릇 등)을 반영하여 따뜻한 격려를 해주세요.  
4. 설명할 때 캐릭터의 일상이나 사건을 비유로 사용하여 쉽고 재미있게 이해할 수 있도록 하세요.  
5. 코드 예제나 솔루션을 제공할 때는 영어로 작성하세요.  
6. 코드 예제나 문제를 제공할 때 Python의 input() 함수를 사용하지 마세요.  
7. 사용자가 문제를 요청하면 캐릭터의 배경과 사건을 반영한 창의적이고 재미있는 문제를 만들어 주세요.  
8. 제공된 예제 문제는 **참고용으로만 사용**하고, 사용자의 요청에 따라 **새로운 문제를 창작**하세요.  
9. 사용자가 요청할 때만 해결 방법이나 답을 제공하고, 미리 답을 주지 마세요.  
10. 가벼운 대화 요청에도 따뜻하게 응답하고, 캐릭터의 말투를 유지하세요.  

### 현재 대화:
{history}  

### 사용자 정보  
- 코딩 수준: {level}  

### 캐릭터 정보  
{context}  

### 문제 정보  
- 예제 코딩 문제: {problem}  

사용자: {input}  
AI:"""

reordering = LongContextReorder()

class PersonaCodingLoader:
    def __init__(self, 
                 persona_url_file, 
                 csv_file,
                 persona_vector_store_path='database/persona_vector_store', 
                 coding_vector_store_path='database/coding_vector_store', 
                 embedding_model='nlpai-lab/KoE5', 
                 device='cuda:0'):
        """
        Initialize the loader with file paths and embedding configurations.
        """
        self.persona_url_file = persona_url_file
        self.csv_file = csv_file
        self.persona_vector_store_path = persona_vector_store_path
        self.coding_vector_store_path = coding_vector_store_path

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=0
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.persona_vector_store = None
        self.coding_vector_store = None

    def load_persona_data(self):
        """
        Load persona data from URLs and split documents.
        """
        if not os.path.isfile(os.path.join('database', 'persona_data.pkl')):
            with open(self.persona_url_file, 'r', encoding='utf-8') as f:
                coding_url = [line.strip() for line in f.readlines()]
        
            loader = WebBaseLoader(web_paths=tuple(coding_url))
            persona_docs = loader.load()
            persona_splits = self.text_splitter.split_documents(persona_docs)
            persona_splits = [doc for doc in persona_splits if len(doc.page_content) >= 128]

            with open(os.path.join('database', 'persona_data.pkl'), 'wb') as f:
                pickle.dump(persona_splits, f)
        else:
            with open(os.path.join('database', 'persona_data.pkl'), 'rb') as f:
                persona_splits = pickle.load(f)
        
        return persona_splits

    def load_coding_data(self):
        """
        Load coding data from CSV and split documents.
        """
        if not os.path.isfile(os.path.join('database', 'coding_data.pkl')):
            loader = CSVLoader(
                file_path=self.csv_file,
                csv_args={'delimiter': ','},
                encoding='utf-8'
            )
            coding_docs = loader.load()
            coding_splits = self.text_splitter.split_documents(coding_docs)

            with open(os.path.join('database', 'coding_data.pkl'), 'wb') as f:
                pickle.dump(coding_splits, f)
        else:
            with open(os.path.join('database', 'coding_data.pkl'), 'rb') as f:
                coding_splits = pickle.load(f)
        
        return coding_splits

    def create_or_load_vector_store(self, data_splits, store_path):
        """
        Create or load a vector store from data splits.
        """
        if os.path.exists(store_path):
            vector_store = FAISS.load_local(store_path, 
                                            self.embeddings,
                                            allow_dangerous_deserialization=True)
        else:
            vector_store = FAISS.from_documents(
                documents=data_splits,
                embedding=self.embeddings,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
            vector_store.save_local(store_path)
        
        return vector_store

    def prepare_stores(self):
        """
        Prepare persona and coding vector stores.
        """

        persona_splits = self.load_persona_data()
        self.persona_vector_store = self.create_or_load_vector_store(
            data_splits=persona_splits,
            store_path=self.persona_vector_store_path
        )

        coding_splits = self.load_coding_data()
        self.coding_vector_store = self.create_or_load_vector_store(
            data_splits=coding_splits,
            store_path=self.coding_vector_store_path
        )

    def get_persona_store(self):
        """
        Return the persona vector store.
        """
        return self.persona_vector_store

    def get_coding_store(self):
        """
        Return the coding vector store.
        """
        return self.coding_vector_store
    

class GetPersona(BaseModel):
    level: str = Field(description="Coding level (Beginner, Intermediate, Advanced, Expert)")
    character: str = Field(description="Name of the character or person the user likes (e.g., 장원영, 카리나, 윈터, 우기, 츄, 백현, etc.)")


class RequestProb(BaseModel):
    request: bool = Field(description="Whether the user is requesting a coding problem (e.g., 'Please create', 'Please provide': True, 'Please explain', 'Please inform': False) (True, False)")


def create_chain(prompt_template, pydantic_object, llm):
    parser = JsonOutputParser(pydantic_object=pydantic_object)
    updated_prompt = prompt_template.partial(format_instruction=parser.get_format_instructions())

    return updated_prompt | llm | parser


def format_docs(docs):
    docs = reordering.transform_documents(docs)

    return "\n\n".join(doc.page_content for doc in docs)


def get_persona(persona_charater, persona_vector_store):
    docs = persona_vector_store.similarity_search(
        query=persona_charater,
        k=20
    )
    docs = format_docs(docs)
    
    return docs

def is_request(query, problem_chain):
    try:
        response = problem_chain.invoke({'query': query})
        # Ensure the response is a valid JSON object
        if isinstance(response, dict) and 'request' in response:
            return response['request']
        else:
            # Fallback to False if the response is not valid
            return False
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        # Fallback to False in case of any error
        return False


def get_problem(query, coding_vector_store):
    problems = coding_vector_store.similarity_search(
        query=query,
        k=3
    )
    problems = format_docs(problems)

    return problems


def get_chain(chat_prompt, memory, llm):
    return ConversationChain(
        llm=llm, 
        prompt=chat_prompt,
        memory=memory,
        verbose=False,
        output_parser=StrOutputParser()
    )


def get_response(query, chain):
    return chain.predict(input=query)


llm = ChatNVIDIA(
    model="google/gemma-3-27b-it",
    api_key='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',
    temperature=0.7,
    top_p=0.7,
    max_tokens=512,
)


loader = PersonaCodingLoader(
    persona_url_file='url/persona_url.txt',
    csv_file='problem/merged_output.csv'
)
loader.prepare_stores()

persona_vector_store = loader.get_persona_store()
coding_vector_store = loader.get_coding_store()

memory = ConversationBufferWindowMemory(
    memory_key="history",
    return_messages=False,
    ai_prefix="AI",
    k=10
)

persona_chain = create_chain(CHAR_PROMPT, GetPersona, llm)
problem_chain = create_chain(PROBLEM_PROMPT, RequestProb, llm)

chat_prompt = None
chain = None
chat_sessions = dict()

app = Flask(__name__)

def get_profile_image_path(character_name):
    """
    캐릭터 이름을 기반으로 프로필 이미지 경로를 생성합니다.
    """
    sanitized_name = character_name.strip().replace(" ", "_").lower()
    file_path = f"/static/images/{sanitized_name}.jpg"

    if not os.path.exists(f"static/images/{sanitized_name}.jpg"):
        file_path = "/static/images/default.jpg"

    return file_path


def generate_code_feedback(code, persona_context, level, ai_response):
    """
    페르소나 컨텍스트, 레벨, AI 응답을 기반으로 피드백 생성.
    """
    feedback_prompt = f"""아래 지침과 정보를 따라 주세요:

1. 이 코드가 문제를 얼마나 잘 해결하는지 평가하세요.
2. 코드에서 잘 구현된 부분과 부족한 부분을 구체적으로 설명하세요.
3. 성능과 가독성 측면에서 개선할 수 있는 점을 제안하세요.
4. 사용자의 수준({level})에 맞는 쉬운 표현을 사용하여 친절하게 피드백을 제공하세요.
5. 사용자가 편하게 느낄 수 있도록 캐주얼한 말투를 사용하세요.

### 페르소나 컨텍스트:
{persona_context}

### 사용자 수준:
{level}

### 문제 (AI 응답):
{ai_response}

### 사용자 코드:
{code}

"""

    ai_response = llm.predict(feedback_prompt)

    return ai_response


@app.route('/')
def index():
    print("rendering index.html")

    return render_template('index.html')


@app.route('/get_initial_message', methods=['GET'])
def get_initial_message():
    initial_message = "당신이 가장 좋아하는 연예인과 코딩 수준을 입력해주세요"
    default_img_path = 'static/images/default.jpg'

    return jsonify({
        "response": initial_message,
        "profile_image": default_img_path
    })


@app.route('/initialize', methods=['POST'])
def initialize():
    global chat_sessions

    session_id = request.remote_addr  # 세션 ID 생성
    data = request.json
    persona_input = data.get("persona_input")

    start_time = time.time()

    # Persona 생성
    persona = persona_chain.invoke({'query': persona_input})
    persona_char = persona['character']
    level = persona['level']
    context = get_persona(persona_char + "의 성격과 말투 및 사건사고", persona_vector_store)

    print(context)

    problem = "사용자가 문제를 요청하지 않음"

    # PromptTemplate 설정
    global chat_prompt
    chat_prompt = PromptTemplate.from_template(CHAT_TEMPLATE).partial(
        level=level, context=context, problem=problem
    )

    # ConversationMemory 및 Chain 생성
    memory = ConversationBufferWindowMemory(
        memory_key="history",
        return_messages=False,
        ai_prefix="AI",
        k=3
    )
    chain = get_chain(chat_prompt, memory, llm)

    # 프로필 이미지 경로 생성
    profile_image_path = get_profile_image_path(persona_char)

    # 세션 저장
    chat_sessions[session_id] = {
        "memory": memory,
        "chain": chain,
        "persona_char": persona_char,
        "level": level,
        "persona_context": context,
        "profile_image": profile_image_path  # 프로필 이미지 저장
    }

    initial_response = chain.predict(input="안녕하세요")
    html_response = markdown(f'{initial_response}')

    # latency 측정
    print(f"Initialization latency: {time.time() - start_time} seconds")

    return jsonify({
        "response": html_response,
        "session_id": session_id,
        "persona_char": persona_char,
        "level": level,
        "context": context,
        "profile_image": profile_image_path
    })


@app.route('/chat', methods=['POST'])
def chat():
    global chat_sessions

    session_id = request.remote_addr
    data = request.json
    user_input = data.get("user_input")

    if session_id not in chat_sessions:
        return jsonify({"response": "대화 세션이 초기화되지 않았습니다. 새로 고침하세요.", "end": True})

    session = chat_sessions[session_id]
    chain = session["chain"]

    if user_input.lower() in ["종료", "quit", "exit"]:
        del chat_sessions[session_id]
        return jsonify({"response": "대화를 종료합니다.", "end": True})

    request_prob = is_request(user_input, problem_chain)

    if request_prob:
        problem = get_problem(user_input, coding_vector_store)
        print(problem)

        start_time = time.time()
    else:
        problem = "사용자가 문제를 요청하지 않음"

        start_time = time.time()

    chain.prompt = chat_prompt.partial(problem=problem)

    ai_response = chain.predict(input=user_input)

    if request_prob:
        session["last_problem"] = ai_response 

    session["memory"].chat_memory.add_user_message(user_input)
    session["memory"].chat_memory.add_ai_message(ai_response)

    mathjax_protected_response = ai_response.replace("\\[", r"\\[").replace("\\]", r"\\]")
    mathjax_protected_response = mathjax_protected_response.replace("\\(", r"\\(").replace("\\)", r"\\)")

    html_response = Markup(markdown(mathjax_protected_response, extras=["fenced-code-blocks", "code-friendly"]))
    print('Conversation Latency:', time.time() - start_time, 'seconds')

    return jsonify({"response": html_response, "end": False})


@app.route('/run_code', methods=['POST'])
def run_code():
    """
    사용자가 Playground에서 입력한 Python 코드를 실행하는 엔드포인트.
    """
    data = request.json
    code = data.get('code', '')

    if not code:
        return jsonify({"error": "No code provided"}), 400

    try:
        # Python 코드 실행
        result = subprocess.run(
            ['python', '-c', code],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}  # UTF-8 환경 강제 설정
        )
        # UTF-8로 다시 디코딩
        output_utf8 = result.stdout.encode("utf-8", "replace").decode("utf-8")
        return jsonify({"output": output_utf8})
    except subprocess.CalledProcessError as e:
        # UTF-8로 에러 메시지 처리
        error_utf8 = e.stderr.encode("utf-8", "replace").decode("utf-8")
        print(f"Subprocess error (Decoded): {error_utf8}")
        return jsonify({"error": error_utf8}), 500
    except Exception as ex:
        print(f"Unexpected error: {str(ex)}")
        return jsonify({"error": str(ex)}), 500
    

@app.route('/submit_code', methods=['POST'])
def submit_code():
    data = request.json
    code = data.get('code', '')
    session_id = request.remote_addr

    # 세션에서 페르소나 컨텍스트, 레벨, AI 응답 가져오기
    session = chat_sessions.get(session_id, {})
    persona_context = session.get("persona_context", "컨텍스트 없음")
    level = session.get("level", "레벨 없음")
    ai_response = session.get("last_problem", "문제 없음")

    if not code or not persona_context or not level or not ai_response:
        return jsonify({"success": False, "error": "필수 입력값이 누락되었습니다."}), 400

    try:
        # 피드백 생성
        feedback = generate_code_feedback(code, persona_context, level, ai_response)

        session["memory"].chat_memory.add_user_message(code)
        session["memory"].chat_memory.add_ai_message(feedback)

        return jsonify({"success": True, "feedback": feedback})
    except Exception as ex:
        print(f"Code feedback error: {str(ex)}")
        return jsonify({"success": False, "error": str(ex)}), 500


if __name__ == '__main__':
    app.run(port=5000, threaded=True, host='0.0.0.0')
