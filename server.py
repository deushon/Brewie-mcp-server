from mcp.server.fastmcp import FastMCP
from typing import List, Any, Optional
from pathlib import Path
import time
import os
import rospy
import base64
import cv2
from datetime import datetime
import numpy as np

from together import Together
import base64

# ROS message types
from std_msgs.msg import Float64, String
from sensor_msgs.msg import Joy, CompressedImage, Image
from cv_bridge import CvBridge

LLMclient = Together()

# Initialize ROS node
rospy.init_node('brewie_mcp_server', anonymous=True)

# Publishers
pan_pub = rospy.Publisher('/head_pan_controller/command', Float64, queue_size=1)
tilt_pub = rospy.Publisher('/head_tilt_controller/command', Float64, queue_size=1)
joy_pub = rospy.Publisher('/joy', Joy, queue_size=1)
action_pub = rospy.Publisher('/app/set_action', String, queue_size=1)

# Subscribers will be created in the CameraSubscriber class
image_topic_name = '/camera/image_raw/compressed'
actionlist_topic_name = "/action_groups_data"

# CV Bridge for image conversion
bridge = CvBridge()


class CameraSubscriber:
    def __init__(self, topic_name):
        self.last_image = None
        self.topic_name = topic_name
        self.subscriber = None

    def on_image_received(self, message):
        # Колбэк, который вызывается при получении нового сообщения
        self.last_image = message

    def get_last_image(self):
        # Метод, который возвращает последний сохраненный снимок
        return self.last_image
    
    def subs(self):
        self.subscriber = rospy.Subscriber(self.topic_name, CompressedImage, self.on_image_received, queue_size=1)

Csubscriber = CameraSubscriber(image_topic_name)

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
    
    # Create Joy message
    message = Joy()
    message.axes = [right_left, forward_backward, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    message.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    joy_pub.publish(message)

    # Create stop message
    message_to_stop = Joy()
    message_to_stop.axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    message_to_stop.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    joy_pub.publish(message_to_stop)

    return "one step!"

@mcp.tool(description="This tool allows you to defend yourself from your opponents. Call it to protect me from opponent. One call to one opponent. I will tell you where the enemy is in relation to you" \
"Tool uses 2 float params (write it without "") for description opponent's position [rotate] were 1.2 is maximum of right -1.2 maximum left [UPDOWN] where -0.3 is maximum down, 0.2 is maximum UP ")
def defend(rotate: float, UPDOWN: float):
    # Clamp values between -1.0 and 1.0
    rotate_fx = max(-1.2, min(1.2, rotate))
    UPDOWN_fx = max(-0.3, min(0.2, UPDOWN))

    # Create Float64 messages for pan and tilt
    panmsg = Float64()
    panmsg.data = rotate_fx

    tiltmsg = Float64()
    tiltmsg.data = UPDOWN_fx

    headZeroMsg = Float64()
    headZeroMsg.data = 0.0

    # Create Joy messages
    defStarmsg = Joy()
    defStarmsg.axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    defStarmsg.buttons = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    defEndmsg = Joy()
    defEndmsg.axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    defEndmsg.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    pan_pub.publish(panmsg)    
    tilt_pub.publish(tiltmsg)
    time.sleep(0.8)
    joy_pub.publish(defStarmsg)
    time.sleep(1.2)
    joy_pub.publish(defEndmsg)
    time.sleep(0.1)
    pan_pub.publish(headZeroMsg)
    time.sleep(0.1)
    tilt_pub.publish(headZeroMsg)
    time.sleep(0.5)

    return "one less threat!"





@mcp.tool(description='This tool getting action from topic on robot and write on python dict[file_name, description]')
def get_available_actions():
    global actions_groups_data  # Needed to modify the global variable
    actions_groups_data = None  # Reset before use
    
    def on_action_received(msg):
        global actions_groups_data
        actions_groups_data = msg.data

    # Create subscriber
    actionlist_sub = rospy.Subscriber(actionlist_topic_name, String, on_action_received)

    start_time = time.time()
    while actions_groups_data is None and (time.time() - start_time) < 5:
        time.sleep(0.1)

    # Unsubscribe
    actionlist_sub.unregister()
    
    if actions_groups_data:
        # Parse the string data as JSON if it's in JSON format
        import json
        try:
            parsed_data = json.loads(actions_groups_data)
            return list(parsed_data.items())  # Convert dict to list of tuples
        except:
            # If not JSON, return as single item
            return [(actions_groups_data, "action")]
    else:
        return []

@mcp.tool(description="This tool run action")
def run_action(action_name: str):
    message = String()
    message.data = action_name

    action_pub.publish(message)
    return "Action published: " + action_name

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
        if hasattr(msg, 'format') and hasattr(msg, 'data'):
            # Обработка сжатого изображения (CompressedImage)
            
            # Декодируем данные Base64
            data_b64 = msg.data
            image_bytes = base64.b64decode(data_b64)
            
            # Преобразуем массив байтов в NumPy-массив
            img_np = np.frombuffer(image_bytes, np.uint8)
            
            # Декодируем изображение из JPEG/PNG с помощью OpenCV
            img_cv = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
            
            if img_cv is None:
                print(f"[Image] Failed to decode image with OpenCV.")
                return "Decoding error"

        elif hasattr(msg, 'height') and hasattr(msg, 'width') and hasattr(msg, 'encoding'):
            # Обработка несжатого изображения (Image)
            height = msg.height
            width = msg.width
            encoding = msg.encoding
            data_b64 = msg.data
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

    # Create Float64 messages for pan positions
    fmsg = []
    
    msg1 = Float64()
    msg1.data = 1.2
    fmsg.append(msg1)

    msg2 = Float64()
    msg2.data = 0.0
    fmsg.append(msg2)

    msg3 = Float64()
    msg3.data = -1.2
    fmsg.append(msg3)

    # Create Joy messages
    defStarmsg = Joy()
    defStarmsg.axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    defStarmsg.buttons = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    defEndmsg = Joy()
    defEndmsg.axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    defEndmsg.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    pan_pub.publish(fmsg[0])    
    time.sleep(0.65)
    get_image()
    pan_pub.publish(fmsg[1])    
    time.sleep(0.65)
    get_image()
    pan_pub.publish(fmsg[2])    
    time.sleep(0.65)
    get_image()
    pan_pub.publish(fmsg[1])


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

    pan_pub.publish(fmsg[int(respons.choices[0].message.content)])    
    time.sleep(0.5)
       
    

    joy_pub.publish(defStarmsg)
    time.sleep(1.2)
    joy_pub.publish(defEndmsg)
    time.sleep(1)    
    pan_pub.publish(fmsg[1])
    time.sleep(0.1) 

    return "Sniper action completed" 

if __name__ == "__main__":
    # Initialize camera subscriber
    Csubscriber.subs()
    
    # Wait for ROS to be ready
    time.sleep(0.5)
    
    # Run MCP server
    mcp.run(transport="streamable-http")
