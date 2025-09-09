from mcp.server.fastmcp import FastMCP
from typing import List, Any, Optional
from pathlib import Path
import time
import os
import roslibpy
import base64
import cv2
from datetime import datetime
import numpy as np
import qrcode
from pyzbar import pyzbar
import json
import requests
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solana.rpc.commitment import Commitment
from solders.transaction import Transaction
from solders.keypair import Keypair
from solders.pubkey import Pubkey as PublicKey
from solders.system_program import TransferParams, transfer
from solders.message import Message
from solders.hash import Hash
from solders.address_lookup_table_account import AddressLookupTableAccount
import base58

from together import Together
import base64


LLMclient = Together()
ROSclient = roslibpy.Ros(host='localhost', port=9090)

pan = roslibpy.Topic(ROSclient, '/head_pan_controller/command', 'std_msgs/Float64')
tilt = roslibpy.Topic(ROSclient, '/head_tilt_controller/command', 'std_msgs/Float64')
joy = roslibpy.Topic(ROSclient, '/joy', 'sensor_msgs/Joy')
image_topic = roslibpy.Topic(ROSclient, '/camera/image_raw/compressed', 'sensor_msgs/CompressedImage',queue_size=1,queue_length=1)
actionlist = roslibpy.Topic(ROSclient, "/action_groups_data", "std_msgs/String")
action = roslibpy.Topic(ROSclient, '/app/set_action', 'std_msgs/String')


class CameraSubscriber:
    def __init__(self, Rclient, imTop):
        self.last_image = None
        self.client = Rclient
        self.image_topic = imTop

    def on_image_received(self, message):
        # Колбэк, который вызывается при получении нового сообщения
        self.last_image = message

    def get_last_image(self):
        # Метод, который возвращает последний сохраненный снимок
        return self.last_image
    def subs(self):
        self.image_topic.subscribe(self.on_image_received)

Csubscriber = CameraSubscriber(ROSclient,image_topic)

def get_files_in_directory(directory_path):

  files = []
  for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)
    if os.path.isfile(item_path):
      files.append(item)
  return files

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def photo_cln(folder_path):

    if not os.path.isdir(folder_path):
        print(f"Error: Path '{folder_path}' break")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"DELL: {file_path}")
        except Exception as e:
            print(f"Dell error {file_path}: {e}")

def detect_qr_code(image_path):
    """Обнаруживает QR-код на изображении и возвращает его содержимое"""
    try:
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            return None, "Не удалось загрузить изображение"
        
        # Декодируем QR-коды
        qr_codes = pyzbar.decode(image)
        
        if not qr_codes:
            return None, "QR-код не найден на изображении"
        
        # Возвращаем содержимое первого найденного QR-кода
        qr_data = qr_codes[0].data.decode('utf-8')
        return qr_data, "QR-код успешно распознан"
        
    except Exception as e:
        return None, f"Ошибка при распознавании QR-кода: {str(e)}"

def validate_sol_address(address):
    """Проверяет, является ли адрес валидным адресом Solana"""
    try:
        # Базовая проверка длины адреса Solana (44 символа в base58)
        if len(address) != 44:
            return False, "Неверная длина адреса Solana"
        
        # Проверяем, что адрес содержит только допустимые символы base58
        import base58
        try:
            decoded = base58.b58decode(address)
            if len(decoded) != 32:  # Solana адреса должны декодироваться в 32 байта
                return False, "Неверный формат адреса Solana"
        except:
            return False, "Неверный формат адреса Solana"
        
        return True, "Адрес Solana валиден"
    except Exception as e:
        return False, f"Ошибка валидации адреса: {str(e)}"

def load_private_key():
    """Загружает закрытый ключ из файла master_sh"""
    try:
        key_file = "master_sh/sol_private_key"
        if not os.path.exists(key_file):
            return None, "Файл с закрытым ключом не найден"
        
        with open(key_file, 'r') as f:
            private_key = f.read().strip()
        
        return private_key, "Закрытый ключ загружен"
    except Exception as e:
        return None, f"Ошибка загрузки закрытого ключа: {str(e)}"

def transfer_sol(to_address, amount, private_key):
    """Выполняет реальный перевод SOL в сети Solana"""
    try:
        print(f"Начинаем перевод {amount} SOL на адрес {to_address}")
        
        # Подключаемся к Solana RPC (mainnet)
        rpc_url = "https://api.mainnet-beta.solana.com"
        client = Client(rpc_url)
        
        # Создаем Keypair из закрытого ключа
        try:
            # Декодируем закрытый ключ из base58
            private_key_bytes = base58.b58decode(private_key)
            keypair = Keypair.from_bytes(private_key_bytes)
            print(f"Кошелек загружен: {keypair.pubkey()}")
        except Exception as e:
            return False, f"Ошибка загрузки закрытого ключа: {str(e)}"
        
        # Создаем PublicKey для получателя
        try:
            recipient_pubkey = PublicKey.from_string(to_address)
        except Exception as e:
            return False, f"Неверный адрес получателя: {str(e)}"
        
        # Конвертируем SOL в lamports (1 SOL = 1,000,000,000 lamports)
        lamports = int(amount * 1_000_000_000)
        
        # Получаем последний блок хеш
        try:
            recent_blockhash = client.get_latest_blockhash()
            if recent_blockhash.value is None:
                return False, "Не удалось получить последний блок хеш"
        except Exception as e:
            return False, f"Ошибка получения блок хеша: {str(e)}"
        
        # Создаем транзакцию
        try:
            # Создаем инструкцию перевода
            transfer_instruction = transfer(
                TransferParams(
                    from_pubkey=keypair.pubkey(),
                    to_pubkey=recipient_pubkey,
                    lamports=lamports
                )
            )
            
            # Создаем сообщение транзакции
            message = Message.new_with_blockhash(
                instructions=[transfer_instruction],
                payer=keypair.pubkey(),
                blockhash=Hash.from_string(str(recent_blockhash.value.blockhash))
            )
            
            # Создаем транзакцию
            transaction = Transaction.new_unsigned(message)
            
        except Exception as e:
            return False, f"Ошибка создания транзакции: {str(e)}"
        
        # Подписываем транзакцию
        try:
            transaction.sign([keypair], Hash.from_string(str(recent_blockhash.value.blockhash)))
            print("Транзакция подписана")
        except Exception as e:
            return False, f"Ошибка подписания транзакции: {str(e)}"
        
        # Отправляем транзакцию
        try:
            print("Отправляем транзакцию в сеть...")
            result = client.send_transaction(
                transaction,
                opts=TxOpts(skip_preflight=False, preflight_commitment="confirmed")
            )
            
            if result.value is None:
                return False, "Транзакция не была отправлена"
            
            signature = result.value
            print(f"Транзакция отправлена: {signature}")
            
            # Ждем подтверждения
            print("Ждем подтверждения транзакции...")
            confirmation = client.confirm_transaction(signature, commitment="confirmed")
            
            # Проверяем статус подтверждения
            confirmation_status = confirmation.value[0].confirmation_status
            print(f"Статус подтверждения: {confirmation_status}")
            print(f"Тип статуса: {type(confirmation_status)}")
            
            # Проверяем, что транзакция подтверждена
            # Статус может быть строкой или enum объектом
            status_str = str(confirmation_status).lower()
            is_confirmed = (
                "confirmed" in status_str or 
                "finalized" in status_str or
                confirmation_status == "confirmed" or
                confirmation_status == "finalized"
            )
            
            if is_confirmed:
                return True, f"Перевод {amount} SOL на адрес {to_address} выполнен успешно! Подпись транзакции: {signature}"
            else:
                return False, f"Транзакция не подтверждена. Статус: {confirmation_status}"
                
        except Exception as e:
            return False, f"Ошибка отправки транзакции: {str(e)}"
        
    except Exception as e:
        return False, f"Критическая ошибка при выполнении перевода: {str(e)}"


mcp = FastMCP("ros-mcp-server")
actions_groups_data: dict[str, str] = None


@mcp.tool(description="This tool makes a robot move by one step in any direction." \
"Tool uses joystick emulate [z][x] -1.0 for right, 1.0 for left, -1.0 for backward, 1.0 for forward")
def make_step(x: float, z: float):
    # Validate input
    right_left = x
    forward_backward = z
    
    # Clamp values between -1.0 and 1.0
    right_left = max(-1.0, min(1.0, right_left))
    forward_backward = max(-1.0, min(1.0, forward_backward))
    
    message = {
        'axes': [right_left, forward_backward, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'buttons': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    joy.publish(message)

    message_to_stop = {
        'axes': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'buttons': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    joy.publish(message_to_stop)

    return "one step!"

@mcp.tool(description="This tool allows you to defend yourself from your opponents. Call it to protect me from opponent. One call to one opponent. I will tell you where the enemy is in relation to you" \
"Tool uses 2 float params (write it without "") for description opponent's position [rotate] were 1.2 is maximum of right -1.2 maximum left [UPDOWN] where -0.3 is maximum down, 0.2 is maximum UP ")
def defend(rotate: float, UPDOWN: float):
    # Clamp values between -1.0 and 1.0
    rotate_fx = max(-1.2, min(1.2, rotate))
    UPDOWN_fx = max(-0.3, min(0.2, UPDOWN))

    panmsg = roslibpy.Message({
        'position': rotate_fx,
        'duration': 0.5,
    })

    tiltmsg = roslibpy.Message({
        'position': UPDOWN_fx,
        'duration': 0.5,
    })


    headZeroMsg = roslibpy.Message({
        'position': 0,
        'duration': 0.5,
    })

    defStarmsg = roslibpy.Message({
        'axes': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'buttons': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })

    defEndmsg = roslibpy.Message({
        'axes': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'buttons': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })

    pan.publish(panmsg)    
    tilt.publish(tiltmsg)
    time.sleep(0.8)
    joy.publish(defStarmsg)
    time.sleep(1.2)
    joy.publish(defEndmsg)
    time.sleep(0.1)
    pan.publish(headZeroMsg)
    time.sleep(0.1)
    tilt.publish(headZeroMsg)
    time.sleep(0.5)

    #joy.unadvertise()
    #tilt.unadvertise()
    #pan.unadvertise()


    return "one less threat!"





@mcp.tool(description='This tool getting action from topic on robot and write on python dict[file_name, description]')
def get_available_actions():
    global actions_groups_data  # Needed to modify the global variable
    actions_groups_data = None  # Reset before use
    
    
    def on_action_received(msg):
        global actions_groups_data
        actions_groups_data = msg

    actionlist.subscribe(on_action_received)

    start_time = time.time()
    while actions_groups_data is None and (time.time() - start_time) < 5:
        time.sleep(0.1)

    actionlist.unsubscribe()
    

    if actions_groups_data:
        return list(actions_groups_data.items())  # Convert dict to list of tuples
    else:
        return []

@mcp.tool(description="This tool run action")
def run_action(action_name: str):

    message = ({
        'data': action_name
    })

    return action.publish(message)

@mcp.tool(description="This tool used to get raw image from robot and save on user pc on directory like downloads")
def get_image():
    #TODO IN sniper game back images on 1 side only. I thn what it error from subscriber
    try:
        # Получаем последнее сообщение.
        received_msg = Csubscriber.get_last_image()

        start_time = time.time()
        while received_msg is None and (time.time() - start_time) < 5:
            time.sleep(0.1)

        if received_msg is None:
            print("[Image] No data received from subscriber")
            return "No data"

        msg = received_msg

        # Проверяем, является ли формат сжатым.
        if 'format' in msg and 'data' in msg:
            # Обработка сжатого изображения (CompressedImage)
            
            # Декодируем данные Base64
            data_b64 = msg['data']
            image_bytes = base64.b64decode(data_b64)
            
            # Преобразуем массив байтов в NumPy-массив
            img_np = np.frombuffer(image_bytes, np.uint8)
            
            # Декодируем изображение из JPEG/PNG с помощью OpenCV
            img_cv = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
            
            if img_cv is None:
                print(f"[Image] Failed to decode image with OpenCV.")
                return "Decoding error"

        elif 'height' in msg and 'width' in msg and 'encoding' in msg:
            # Обработка несжатого изображения (Image)
            height = msg["height"]
            width = msg["width"]
            encoding = msg["encoding"]
            data_b64 = msg["data"]
            image_bytes = base64.b64decode(data_b64)
            img_np = np.frombuffer(image_bytes, dtype=np.uint8)

            if encoding == "rgb8":
                img_np = img_np.reshape((height, width, 3))
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif encoding == "bgr8":
                img_cv = img_np.reshape((height, width, 3))
            elif encoding == "mono8":
                img_cv = img_np.reshape((height, width))
            else:
                print(f"[Image] Unsupported encoding: {encoding}")
                return "Format error"
        else:
            print("[Image] Unsupported message format.")
            return "Format error"

        downloads_dir = "photos/environment"
        # Убедимся, что директория существует.
        os.makedirs(downloads_dir, exist_ok=True)
        items = os.listdir(downloads_dir)

        timestamp = str(len(items))
        save_path = os.path.join(downloads_dir, f"image_{timestamp}.png")
        cv2.imwrite(str(save_path), img_cv)

        print(f"[Image] Saved to {save_path}")

        return img_cv

    except Exception as e:
        print(f"[Image] Failed to receive or decode: {e}")
        return "Failure"

    
@mcp.tool(description="This tool allows you to play sniper unlike the defender tool here the person says the description of the target and not its position, where it is the robot decides itself" \
"Tool use one string param, it is description of target to shoot")
def sniper(targediscr:str):

    print("startsnipet tool")
    photo_cln("photos/environment")

    pan = roslibpy.Topic(ROSclient, '/head_pan_controller/command', 'std_msgs/Float64')
    joy = roslibpy.Topic(ROSclient, '/joy', 'sensor_msgs/Joy')

    fmsg=[]

    fmsg.append(roslibpy.Message({
        'position': 1.2,
        'duration': 0.3,
    }))


    fmsg.append(roslibpy.Message({
        'position': 0,
        'duration': 0.3,
    }))

    fmsg.append(roslibpy.Message({
        'position': -1.2,
        'duration': 0.3,
    }))



    defStarmsg = roslibpy.Message({
        'axes': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'buttons': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })

    defEndmsg = roslibpy.Message({
        'axes': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'buttons': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })

    pan.publish(fmsg[0])    
    time.sleep(0.65)
    get_image()
    pan.publish(fmsg[1])    
    time.sleep(0.65)
    get_image()
    pan.publish(fmsg[2])    
    time.sleep(0.65)
    get_image()
    pan.publish(fmsg[1])


    getDescriptionPrompt = "You see 3 photos (0,1,2). Return only the number of the photo in which, in your opinion, the object most closely resembles " + targediscr + ". The answer should only be one digit without additional words."

    images = ["image_0.png","image_1.png","image_2.png"]

    base64_images = []
    for img in images:
        base64_images.append(encode_image("photos/environment/"+img))

    respons = LLMclient.chat.completions.create(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": getDescriptionPrompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[0]}"
                }
            },
                        {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[1]}"
                }
            },
                        {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[2]}"
                }
            }
        ]
    }],
    )

    pan.publish(fmsg[int(respons.choices[0].message.content)])    
    time.sleep(0.5)
       
    

    joy.publish(defStarmsg)
    time.sleep(1.2)
    joy.publish(defEndmsg)
    time.sleep(1)    
    pan.publish(fmsg[1])
    time.sleep(0.1) 

    joy.unadvertise()
    pan.unadvertise()
    return 

@mcp.tool(description="This tool performs SOL transfer by taking a photo, detecting QR code with SOL wallet address, and executing the transfer. Takes amount in SOL as parameter.")
def BrewPay(amount: float):
    """
    Выполняет перевод SOL:
    1. Очищает папку с фото
    2. Делает снимок
    3. Ищет и распознает QR-код с адресом SOL кошелька
    4. Валидирует адрес
    5. Выполняет перевод
    """
    try:
        print(f"Начинаем перевод {amount} SOL")
        
        # 1. Очищаем папку с фото
        photo_cln("photos/environment")
        print("Папка с фото очищена")
        
        # 2. Делаем снимок
        print("Делаем снимок...")
        image_result = get_image()
        print("Готов")
        
        image_path = "photos/environment/image_0.png" 
        
        print(f"Анализируем снимок: {image_path}")
        
        # 4. Распознаем QR-код
        qr_data, qr_message = detect_qr_code(image_path)
        if qr_data is None:
            return f"Ошибка: {qr_message}"
        
        print(f"QR-код распознан: {qr_data}")
        
        # 5. Валидируем адрес SOL
        is_valid, validation_message = validate_sol_address(qr_data)
        if not is_valid:
            return f"Ошибка: {validation_message}"
        
        print(f"Адрес SOL валиден: {qr_data}")
        
        # 6. Загружаем закрытый ключ
        private_key, key_message = load_private_key()
        if private_key is None:
            return f"Ошибка: {key_message}"
        
        print("Закрытый ключ загружен")
        
        # 7. Выполняем перевод
        success, transfer_message = transfer_sol(qr_data, amount, private_key)
        if not success:
            return f"Ошибка перевода: {transfer_message}"
        
        return f"Успешно! {transfer_message}"
        
    except Exception as e:
        return f"Критическая ошибка: {str(e)}"

if __name__ == "__main__":
    ROSclient.run()
    time.sleep(0.5)
    Csubscriber.subs()
    mcp.run(transport="streamable-http")
