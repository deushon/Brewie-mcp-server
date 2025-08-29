import argparse
from mcp.server.fastmcp import FastMCP
from typing import List, Any, Optional
from pathlib import Path
import json
from utils.websocket_manager import WebSocketManager
import time
import os
import roslibpy
import base64
import cv2
from datetime import datetime
import numpy as np

from together import Together
import base64


LLMclient = Together()


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


ROSBRIDGE_IP = "127.0.0.1"
ROSBRIDGE_PORT = 9090

mcp = FastMCP("ros-mcp-server")
ws_manager = WebSocketManager(ROSBRIDGE_IP, ROSBRIDGE_PORT)

actions_groups_data: dict[str, str] = None

@mcp.tool()
def get_topics():
    
    ws_manager.connect()
    topic_info = ws_manager.ws.get_topics()
    ws_manager.close()

    if topic_info:
       return list(topic_info)
    else:
        return "No topics found"

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

    ws_manager.send('/joy', 'sensor_msgs/Joy', message)

    message_to_stop = {
        'axes': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'buttons': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    ws_manager.send('/joy', 'sensor_msgs/Joy', message_to_stop)

    return "one step!"

@mcp.tool(description="This tool allows you to defend yourself from your opponents. Call it to protect me from opponent. One call to one opponent. I will tell you where the enemy is in relation to you" \
"Tool uses 2 float params (write it without "") for description opponent's position [rotate] were 1.2 is maximum of right -1.2 maximum left [UPDOWN] where -0.3 is maximum down, 0.2 is maximum UP ")
def defend(rotate: float, UPDOWN: float):
    # Clamp values between -1.0 and 1.0
    rotate_fx = max(-1.2, min(1.2, rotate))
    UPDOWN_fx = max(-0.3, min(0.2, UPDOWN))

    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    time.sleep(0.5)

    pan = roslibpy.Topic(client, '/head_pan_controller/command', 'std_msgs/Float64')
    tilt = roslibpy.Topic(client, '/head_tilt_controller/command', 'std_msgs/Float64')
    joy = roslibpy.Topic(client, '/joy', 'sensor_msgs/Joy')

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

    joy.unadvertise()
    tilt.unadvertise()
    pan.unadvertise()
    client.terminate()


    return "one less threat!"





@mcp.tool(description='This tool getting action from topic on robot and write on python dict[file_name, description]')
def get_available_actions():
    global actions_groups_data  # Needed to modify the global variable
    actions_groups_data = None  # Reset before use
    
    ws_manager.connect()

    # topic for read
    topic = roslibpy.Topic(
        ws_manager.ws,
        "/action_groups_data",
        "std_msgs/String"
    )

    def on_action_received(msg):
        global actions_groups_data
        actions_groups_data = msg

    topic.subscribe(on_action_received)

    start_time = time.time()
    while actions_groups_data is None and (time.time() - start_time) < 5:
        time.sleep(0.1)

    topic.unsubscribe()
    ws_manager.close()

    if actions_groups_data:
        return list(actions_groups_data.items())  # Convert dict to list of tuples
    else:
        return []

@mcp.tool(description="This tool run action")
def run_action(action_name: str):

    message = ({
        'data': action_name
    })

    return ws_manager.send('app/set_action', 'std_msgs/String', message)

@mcp.tool(description="This tool used to get raw image from robot and save on user pc on directory like downloads")
def get_image():
    ws_manager.connect()

    image_topic = roslibpy.Topic(
        ws_manager.ws,
        '/camera/image_raw',
        'sensor_msgs/Image'
    )
    
    try:
        received_msg = None

        def on_image_received(msg):
            nonlocal received_msg
            received_msg = msg

        image_topic.subscribe(on_image_received)

        start_time = time.time()
        while received_msg is None and (time.time() - start_time) < 5:
            time.sleep(0.1)

        if received_msg is None:
            print("[Image] No data received from subscriber")
            image_topic.unsubscribe()
            return "No data"

        msg = received_msg

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
            image_topic.unsubscribe()
            return "Format error"
        downloads_dir = "photos/environment"
        items = os.listdir(downloads_dir)

        timestamp = str(len(items))
        save_path = downloads_dir +"/"+ f"image_{timestamp}.png"
        #Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img_cv)
        #os.startfile(save_path)
        
        print(f"[Image] Saved to {save_path}")

        image_topic.unsubscribe()
        ws_manager.close()
        return img_cv

    except Exception as e:
        if 'image_topic' in locals():
            image_topic.unsubscribe()
        return "[Image] Failed to receive or decode: "
    
@mcp.tool(description="This tool allows you to play sniper unlike the defender tool here the person says the description of the target and not its position, where it is the robot decides itself" \
"Tool use one string param, it is description of target to shoot")
def sniper(targediscr:str):


    photo_cln("photos/environment")
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    time.sleep(0.5)

    pan = roslibpy.Topic(client, '/head_pan_controller/command', 'std_msgs/Float64')
    joy = roslibpy.Topic(client, '/joy', 'sensor_msgs/Joy')

    fmsg=[]

    fmsg.append(roslibpy.Message({
        'position': 1.2,
        'duration': 0.5,
    }))


    fmsg.append(roslibpy.Message({
        'position': 0,
        'duration': 0.5,
    }))

    fmsg.append(roslibpy.Message({
        'position': -1.2,
        'duration': 0.5,
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
    time.sleep(1)
    get_image()
    time.sleep(1)
    pan.publish(fmsg[1])    
    time.sleep(1)
    get_image()
    time.sleep(1)
    pan.publish(fmsg[2])    
    time.sleep(1)
    get_image()
    time.sleep(1)
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
    time.sleep(0.8)
       
    

    joy.publish(defStarmsg)
    time.sleep(1.2)
    joy.publish(defEndmsg)
    time.sleep(1)    
    pan.publish(fmsg[1])
    time.sleep(0.1) 

    joy.unadvertise()
    pan.unadvertise()
    client.terminate()
    return 

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
