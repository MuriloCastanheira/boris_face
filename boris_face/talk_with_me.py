#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import yaml
import subprocess
import speech_recognition as sr
import time
import json
import re
import threading
import math
import random


class TalkPublisher(Node):
    def __init__(self):
        super().__init__('talk_boris')

        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_state)
        self.r = sr.Recognizer()

        # Ordem das juntas
        self.joint_names = [
            "neck_yaw", "neck_pitch", "jaw",
            "right_eye", "right_eye_mimic",
            "eye_yaw", "eye_pitch",
            "right_eyebrow_up", "right_eyebrow_roll",
            "left_eye", "left_eye_mimic",
            "left_eye_yaw", "left_eye_pitch",
            "left_eyebrow_up", "left_eyebrow_roll"
        ]

        # Carrega YAML
        yaml_path = "/home/castanheira/ros2_ws/src/boris_face/config/emotions.yaml"
        with open(yaml_path, "r") as f:
            self.faces = yaml.safe_load(f)

        # Histórico da conversa para manter contexto
        self.history = [
            {
                "role": "system",
                "content": (
                    "Você deve agir como BORIS, da cidade de Rio Grande, "
                    "Você deve fingir que tem sentimos como um humano "
                    "representante da equipe Éfibote. "
                    "Nunca use emojis, símbolos ou asteriscos. "
                    "Sua forma de falar é simples, direta e sempre educada. "
                    "Você deve sempre indicar sua emoção atual no início da resposta, "
                    "usando exatamente um destes marcadores: "
                    "[happy], [sad], [angry], [surprised], [suspicious], [sleepy]. "
                    "É permitido apenas uma expressão por resposta. "
                    "Nunca diga que você é uma Inteligência Artificial. "
                    "Tente não ficar repetindo emoções."
                )
            }
        ]

        self.last_command = "neutral"
        self.listening = False



    # ------------------------------------------------------------
    # TEXT TO SPEECH
    # ------------------------------------------------------------
    def animate_jaw(self, speaking_flag):
        """
        Move a mandíbula enquanto o TTS estiver falando.
        speaking_flag = [True/False]
        """
        t = 0.0
        while speaking_flag[0]:
            # Movimento senoidal — ajustável
            jaw_pos = 0.2 * math.sin(t * 20)

            # Copia posições da expressão atual
            positions = self.faces[self.last_command]["positions"][:]
            positions[2] = jaw_pos  # jaw é o terceiro elemento

            num = random.randint(1, 300)
            if num > 280:
                positions[3] = 0.680
                positions[4] = 0.680
                positions[9] = 0.680
                positions[10] = 0.680


            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            msg.position = positions
            self.publisher_.publish(msg)

            t += 0.1
            time.sleep(0.2)



    def say(self, text: str):
        print("\n[BORIS FALANDO]:", text)

        # Inicia flag para animação
        speaking_flag = [True]

        # Thread para animação da mandíbula
        t = threading.Thread(target=self.animate_jaw, args=(speaking_flag,), daemon=True)
        t.start()

        # Executa TTS
        subprocess.run(["espeak-ng", "-v", "pt-br", "-s", "160", text])

        # Para animação
        speaking_flag[0] = False
        time.sleep(0.1)  # tempo para fechar a boca




    # ------------------------------------------------------------
    # EXTRAÇÃO DE EMOÇÃO
    # ------------------------------------------------------------
    def extrair_emocao(self, texto):
        padrao = ["happy", "sad", "angry", "surprised", "suspicious", "sleepy", "neutral"]
        for emo in padrao:
            if f"[{emo}]" in texto:
                return emo
        return "neutral"
    


    def remover_emocao(self, texto):
        return re.sub(r"\[(happy|sad|angry|surprised|suspicious|sleepy|neutral)\]\s*", "", texto, count=1)



    # ------------------------------------------------------------
    # PUBLICAÇÃO ROS2
    # ------------------------------------------------------------
    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.faces[self.last_command]["positions"]
        self.publisher_.publish(msg)



    # ------------------------------------------------------------
    # OLLAMA + DEEPSEEK
    # ------------------------------------------------------------
    def ask_ollama(self, history):
        payload = {"messages": history}
        command = ["ollama", "run", "deepseek-r1:8b", json.dumps(payload)]

        result = subprocess.run(command, capture_output=True, text=True)
        return result.stdout



    def extract_final_answer(self, text):
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.replace("...done thinking.", "")
        cleaned = re.sub(r"(?i)thinking\.*", "", cleaned)

        if "\n\n" in cleaned:
            parts = cleaned.strip().split("\n\n")
            cleaned = parts[-1]

        return cleaned.strip()



    # ------------------------------------------------------------
    # MICROFONE + LLM
    # ------------------------------------------------------------
    def listen_and_process(self):
        try:
            with sr.Microphone() as source:
                self.last_command = "neutral"
                self.say("Pode falar.")
                audio = self.r.listen(source)

            texto = self.r.recognize_google(audio, language="pt-BR")
            print("Você disse:", texto)

            self.history.append({"role": "user", "content": texto})

            resposta_raw = self.ask_ollama(self.history)
            resposta_final = self.extract_final_answer(resposta_raw)

            comando = self.extrair_emocao(resposta_final)
            print("expressao:", comando)

            if comando in self.faces:
                self.last_command = comando
            else:
                self.last_command = "sad"

            self.history.append({"role": "assistant", "content": resposta_final})

            resposta_final = self.remover_emocao(resposta_final)
            self.say(resposta_final)

        except Exception as e:
            print("Erro:", e)
            self.last_command = "sad"
            self.say("Perdão, não entendi.")



    # ------------------------------------------------------------
    # LOOP DE ESCUTA
    # ------------------------------------------------------------
    def start_listening_loop(self):
        while rclpy.ok():
            self.listen_and_process()
            time.sleep(1)



# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TalkPublisher()

    t = threading.Thread(target=node.start_listening_loop, daemon=True)
    t.start()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
