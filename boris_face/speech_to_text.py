import speech_recognition as sr

# cria o reconhecedor
r = sr.Recognizer()

# usa o microfone como fonte de áudio
with sr.Microphone() as source:
    print("Fale algo...")
    audio = r.listen(source)

try:
    # reconhece usando o Google Speech Recognition
    texto = r.recognize_google(audio, language="pt-BR")
    print("Você disse:", texto)
except sr.UnknownValueError:
    print("Não entendi o áudio.")
except sr.RequestError:
    print("Erro ao acessar o serviço de reconhecimento.")
