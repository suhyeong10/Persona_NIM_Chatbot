document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-btn');
    const runCodeButton = document.getElementById('run-code-btn'); // Playground 실행 버튼
    const codeOutput = document.getElementById('code-output'); // Playground 출력 영역

    let aiName = ''; // AI 이름
    let aiProfileImage = ''; // AI 프로필 이미지
    let sessionInitialized = false; // 세션 초기화 상태

    // 추가된 변수
    let personaLevel = ''; // 사용자 수준
    let personaContext = ''; // 페르소나 컨텍스트

    // CodeMirror 초기화
    const codeEditorElement = document.getElementById('code-editor');
    const codeMirrorEditor = CodeMirror(codeEditorElement, {
        lineNumbers: true,
        mode: 'python',
        theme: 'monokai',
        indentUnit: 4,
        tabSize: 4,
        indentWithTabs: false,
        lineWrapping: true,
        autofocus: true,
        autoCloseBrackets: {
            pairs: "[]{}()''\"\"",
            triples: "'\"",
            explode: "[]{}()",
            override: true
        }
    });

    // f"" 및 f'' 자동 완성 처리
    codeMirrorEditor.on('beforeChange', function (cm, change) {
        if (change.origin === '+input' && change.text.length === 1) {
            const typedChar = change.text[0];
            const cursor = cm.getCursor();
            const prevCharPos = { line: cursor.line, ch: cursor.ch - 1 };
            const prevChar = cm.getRange(prevCharPos, cursor);

            if ((typedChar === '"' || typedChar === "'") && prevChar === 'f') {
                change.cancel();
                cm.replaceRange(
                    `${prevChar}${typedChar}${typedChar}`,
                    { line: cursor.line, ch: cursor.ch - 1 },
                    cursor
                );
                cm.setCursor({ line: cursor.line, ch: cursor.ch + 1 });
            }
        }
    });

    // Ctrl+ Shift + F10 키 이벤트 리스너 추가
    codeMirrorEditor.on('keydown', function (cm, event) {
        if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'F10') {
            event.preventDefault(); // 기본 동작 방지
            runCode(); // 코드 실행 함수 호출
        }
    });

    // 입력창 자동 높이 조절 함수
    function autoResizeTextarea(element) {
        element.style.height = 'auto';
        element.style.height = element.scrollHeight + 'px';

        if (element.scrollHeight >= parseInt(window.getComputedStyle(element).maxHeight)) {
            element.style.overflowY = 'auto';
        } else {
            element.style.overflowY = 'hidden';
        }
    }

    userInput.addEventListener('input', () => {
        autoResizeTextarea(userInput);
    });

    function renderMathJax() {
        if (window.MathJax) {
            MathJax.typesetPromise()
                .then(() => console.log("MathJax rendering complete."))
                .catch((err) => console.error("MathJax rendering error:", err));
        }
    }

    // 메시지 추가 함수
    function addMessage(content, sender, profileImage = '', name = '') {
        const messageRow = document.createElement('div');
        messageRow.className = `message-row ${sender}`;

        if (profileImage) {
            const profileImg = document.createElement('img');
            profileImg.src = profileImage;
            profileImg.className = 'profile-img';
            messageRow.appendChild(profileImg);
        }

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        if (name) {
            const profileName = document.createElement('div');
            profileName.className = 'message-name';
            profileName.textContent = name;
            messageContent.appendChild(profileName);
        }

        const messageBubble = document.createElement('div');
        messageBubble.className = 'message';

        if (sender === 'ai' && content.includes('```')) {
            const codeBlock = document.createElement('pre');
            const code = document.createElement('code');
            code.className = 'language-python';
            code.textContent = content.replace(/```/g, '').trim();
            codeBlock.appendChild(code);
            messageBubble.appendChild(codeBlock);
        } else {
            messageBubble.innerHTML = content;
        }

        messageContent.appendChild(messageBubble);
        messageRow.appendChild(messageContent);
        chatBox.appendChild(messageRow);
        chatBox.scrollTop = chatBox.scrollHeight;
        hljs.highlightAll();
        renderMathJax();
    }

    function addLoadingMessage() {
        const loadingRow = document.createElement('div');
        loadingRow.className = 'message-row ai'; // AI 스타일 적용
    
        // 프로필 사진 추가 (세션에서 가져오기)
        const profileImg = document.createElement('img');
        profileImg.src = aiProfileImage || 'static/images/default.jpg'; // 세션의 프로필 이미지 사용
        profileImg.className = 'profile-img';
        loadingRow.appendChild(profileImg);
    
        // 메시지 내용 컨테이너
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
    
        // 이름 추가
        const profileName = document.createElement('div');
        profileName.className = 'message-name';
        profileName.textContent = aiName || 'AI'; // 세션의 이름 사용
        messageContent.appendChild(profileName);
    
        // 로딩 애니메이션 (점 3개) 추가
        const messageBubble = document.createElement('div');
        messageBubble.className = 'message';
    
        const loadingBubble = document.createElement('div');
        loadingBubble.className = 'loading-dots';
    
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'dot';
            loadingBubble.appendChild(dot);
        }
    
        messageBubble.appendChild(loadingBubble);
        messageContent.appendChild(messageBubble);
        loadingRow.appendChild(messageContent);
    
        chatBox.appendChild(loadingRow);
    
        // 스크롤 맨 아래로 이동
        chatBox.scrollTop = chatBox.scrollHeight;
    
        return loadingRow; // 로딩 메시지 요소 반환
    }    

    function showInitialMessage() {
        fetch('/get_initial_message')
            .then((response) => response.json())
            .then((data) => {
                addMessage(data.response, 'ai', data.profile_image, 'AI');
            })
            .catch((error) => console.error('초기 메시지 로드 오류:', error));
    }

    function initializeChat(personaInput, loadingMessage) {
        fetch('/initialize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ persona_input: personaInput })
        })
            .then((response) => response.json())
            .then((data) => {
                sessionInitialized = true;
                aiName = data.persona_char;
                aiProfileImage = data.profile_image;

                // 추가된 데이터 저장
                personaLevel = data.level;
                personaContext = data.context;

                chatBox.removeChild(loadingMessage);
                addMessage(data.response, 'ai', aiProfileImage, aiName);
            })
            .catch((error) => {
                console.error('초기화 오류:', error);
                chatBox.removeChild(loadingMessage);
            });
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
    
        // 사용자 메시지를 Markdown에서 HTML로 변환
        if (typeof marked !== 'undefined') {
            const userMessageHtml = marked.parse(message);
            addMessage(userMessageHtml, 'user'); // 사용자 말풍선에 표시
        } else {
            addMessage(message, 'user'); // 일반 텍스트로 표시
        }
    
        userInput.value = '';
        userInput.style.height = 'auto'; // 높이 초기화
        userInput.style.overflowY = 'hidden'; // 스크롤바 숨기기
    
        // 기본 로딩 메시지 추가
        const loadingMessage = addLoadingMessage(sessionInitialized ? false : true);
    
        if (!sessionInitialized) {
            initializeChat(message, loadingMessage); // 초기화 처리
        } else {
            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_input: message })
            })
                .then(response => response.json())
                .then(data => {
                    // 로딩 메시지 제거
                    chatBox.removeChild(loadingMessage);
    
                    // AI 응답 메시지를 Markdown에서 HTML로 변환
                    if (typeof marked !== 'undefined') {
                        const aiResponseHtml = marked.parse(data.response);
                        addMessage(aiResponseHtml, 'ai', aiProfileImage, aiName);
                    } else {
                        addMessage(data.response, 'ai', aiProfileImage, aiName); // 일반 텍스트로 표시
                    }
                    renderMathJax()
                })
                .catch(error => {
                    chatBox.removeChild(loadingMessage);
                    addMessage(`Error: ${error.message}`, 'ai', aiProfileImage, aiName);
                    console.error('Message processing error:', error);
                });
        }
    }

    // Playground 실행 버튼 클릭 이벤트
    runCodeButton.addEventListener('click', () => {
        runCode();
    });

    // 추가: Playground 코드 제출 버튼 처리
    const submitCodeButton = document.getElementById('submit-code-btn');
    submitCodeButton.addEventListener('click', () => {
        submitCode();
    });

    function runCode() {
        const code = codeMirrorEditor.getValue().trim();
        if (!code) {
            codeOutput.textContent = 'Error: No code provided.';
            return;
        }

        fetch('/run_code', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.output) {
                    codeOutput.textContent = data.output;
                } else if (data.error) {
                    codeOutput.textContent = `Error: ${data.error}`;
                } else {
                    codeOutput.textContent = 'Error: Unexpected response from server.';
                }
            })
            .catch((error) => {
                codeOutput.textContent = `Error: ${error.message}`;
                console.error('Error during code execution:', error);
            });
    }

    // 추가된 코드 제출 기능
    function submitCode() {
        const code = codeMirrorEditor.getValue().trim();
        if (!code) {
            addMessage('Error: No code provided.', 'ai'); // 에러 메시지 출력
            return;
        }
    
        // 사용자 코드 메시지를 Markdown에서 HTML로 변환하여 추가
        if (typeof marked !== 'undefined') {
            const userCodeHtml = marked.parse(`\`\`\`python\n${code}\n\`\`\``);
            addMessage(userCodeHtml, 'user'); // 사용자 말풍선에 표시
        } else {
            addMessage('Error: Marked.js is not defined.', 'user');
        }
    
        // 로딩 메시지 추가 (세션의 프로필 이미지를 사용)
        const loadingMessage = addLoadingMessage();
    
        // 서버로 코드 제출
        fetch('/submit_code', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code,
                persona_context: personaContext,
                level: personaLevel
            })
        })
            .then((response) => response.json())
            .then((data) => {
                // 로딩 메시지 제거
                chatBox.removeChild(loadingMessage);
    
                if (data.success) {
                    // AI 피드백을 Markdown에서 HTML로 변환하여 추가
                    if (typeof marked !== 'undefined') {
                        const feedbackHtml = marked.parse(data.feedback);
                        addMessage(feedbackHtml, 'ai', aiProfileImage, aiName);
                    } else {
                        addMessage('Error: Marked.js is not defined.', 'ai', aiProfileImage, aiName);
                    }
                } else if (data.error) {
                    addMessage(`Error: ${data.error}`, 'ai', aiProfileImage, aiName);
                } else {
                    addMessage('Error: Unexpected response from server.', 'ai', aiProfileImage, aiName);
                }
            })
            .catch((error) => {
                // 로딩 메시지 제거
                chatBox.removeChild(loadingMessage);
    
                addMessage(`Error: ${error.message}`, 'ai', aiProfileImage, aiName);
                console.error('Error during code submission:', error);
            });
    }
    
    // Submit Code 버튼 단축키 (Ctrl + Enter)
    document.addEventListener('keydown', (event) => {
        if (event.ctrlKey && event.key === 'Enter') {
            event.preventDefault(); // 기본 동작 방지 (줄바꿈 등)
            submitCode(); // Submit Code 함수 호출
        }
    });


    showInitialMessage();

    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            if (!event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
    });

    sendButton.addEventListener('click', sendMessage);
});
