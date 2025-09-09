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
        # Callback that is called when a new message is received
        self.last_image = message

    def get_last_image(self):
        # Method that returns the last saved image
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
        print(f"Error: Path '{folder_path}' is invalid")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"DELETED: {file_path}")
        except Exception as e:
            print(f"Delete error {file_path}: {e}")

def detect_qr_code(image_path):
    """Detects QR code on image and returns its content"""
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Failed to load image"
        
        # Decode QR codes
        qr_codes = pyzbar.decode(image)
        
        if not qr_codes:
            return None, "QR code not found on image"
        
        # Return content of the first found QR code
        qr_data = qr_codes[0].data.decode('utf-8')
        return qr_data, "QR code successfully recognized"
        
    except Exception as e:
        return None, f"Error recognizing QR code: {str(e)}"

def validate_sol_address(address):
    """Checks if address is a valid Solana address"""
    try:
        # Basic check of Solana address length (44 characters in base58)
        if len(address) != 44:
            return False, "Invalid Solana address length"
        
        # Check that address contains only valid base58 characters
        import base58
        try:
            decoded = base58.b58decode(address)
            if len(decoded) != 32:  # Solana addresses should decode to 32 bytes
                return False, "Invalid Solana address format"
        except:
            return False, "Invalid Solana address format"
        
        return True, "Solana address is valid"
    except Exception as e:
        return False, f"Address validation error: {str(e)}"

def load_private_key():
    """Loads private key from master_sh file"""
    try:
        key_file = "master_sh/sol_private_key"
        if not os.path.exists(key_file):
            return None, "Private key file not found"
        
        with open(key_file, 'r') as f:
            private_key = f.read().strip()
        
        return private_key, "Private key loaded"
    except Exception as e:
        return None, f"Error loading private key: {str(e)}"

def transfer_sol(to_address, amount, private_key):
    """Performs real SOL transfer in Solana network"""
    try:
        print(f"Starting transfer of {amount} SOL to address {to_address}")
        
        # Connect to Solana RPC (mainnet)
        rpc_url = "https://api.mainnet-beta.solana.com"
        client = Client(rpc_url)
        
        # Create Keypair from private key
        try:
            # Decode private key from base58
            private_key_bytes = base58.b58decode(private_key)
            keypair = Keypair.from_bytes(private_key_bytes)
            print(f"Wallet loaded: {keypair.pubkey()}")
        except Exception as e:
            return False, f"Error loading private key: {str(e)}"
        
        # Create PublicKey for recipient
        try:
            recipient_pubkey = PublicKey.from_string(to_address)
        except Exception as e:
            return False, f"Invalid recipient address: {str(e)}"
        
        # Convert SOL to lamports (1 SOL = 1,000,000,000 lamports)
        lamports = int(amount * 1_000_000_000)
        
        # Get latest block hash
        try:
            recent_blockhash = client.get_latest_blockhash()
            if recent_blockhash.value is None:
                return False, "Failed to get latest block hash"
        except Exception as e:
            return False, f"Error getting block hash: {str(e)}"
        
        # Create transaction
        try:
            # Create transfer instruction
            transfer_instruction = transfer(
                TransferParams(
                    from_pubkey=keypair.pubkey(),
                    to_pubkey=recipient_pubkey,
                    lamports=lamports
                )
            )
            
            # Create transaction message
            message = Message.new_with_blockhash(
                instructions=[transfer_instruction],
                payer=keypair.pubkey(),
                blockhash=Hash.from_string(str(recent_blockhash.value.blockhash))
            )
            
            # Create transaction
            transaction = Transaction.new_unsigned(message)
            
        except Exception as e:
            return False, f"Error creating transaction: {str(e)}"
        
        # Sign transaction
        try:
            transaction.sign([keypair], Hash.from_string(str(recent_blockhash.value.blockhash)))
            print("Transaction signed")
        except Exception as e:
            return False, f"Error signing transaction: {str(e)}"
        
        # Send transaction
        try:
            print("Sending transaction to network...")
            result = client.send_transaction(
                transaction,
                opts=TxOpts(skip_preflight=False, preflight_commitment="confirmed")
            )
            
            if result.value is None:
                return False, "Transaction was not sent"
            
            signature = result.value
            print(f"Transaction sent: {signature}")
            
            # Wait for confirmation
            print("Waiting for transaction confirmation...")
            confirmation = client.confirm_transaction(signature, commitment="confirmed")
            
            # Check confirmation status
            confirmation_status = confirmation.value[0].confirmation_status
            print(f"Confirmation status: {confirmation_status}")
            print(f"Status type: {type(confirmation_status)}")
            
            # Check that transaction is confirmed
            # Status can be string or enum object
            status_str = str(confirmation_status).lower()
            is_confirmed = (
                "confirmed" in status_str or 
                "finalized" in status_str or
                confirmation_status == "confirmed" or
                confirmation_status == "finalized"
            )
            
            if is_confirmed:
                return True, f"Transfer of {amount} SOL to target completed successfully! Transaction signature in logs"
            else:
                return False, f"Transaction not confirmed. Status: {confirmation_status}"
                
        except Exception as e:
            return False, f"Error sending transaction: {str(e)}"
        
    except Exception as e:
        return False, f"Critical error during transfer execution: {str(e)}"


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
        # Get the last message.
        received_msg = Csubscriber.get_last_image()

        start_time = time.time()
        while received_msg is None and (time.time() - start_time) < 5:
            time.sleep(0.1)

        if received_msg is None:
            print("[Image] No data received from subscriber")
            return "No data"

        msg = received_msg

        # Check if format is compressed.
        if 'format' in msg and 'data' in msg:
            # Process compressed image (CompressedImage)
            
            # Decode Base64 data
            data_b64 = msg['data']
            image_bytes = base64.b64decode(data_b64)
            
            # Convert byte array to NumPy array
            img_np = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image from JPEG/PNG using OpenCV
            img_cv = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
            
            if img_cv is None:
                print(f"[Image] Failed to decode image with OpenCV.")
                return "Decoding error"

        elif 'height' in msg and 'width' in msg and 'encoding' in msg:
            # Process uncompressed image (Image)
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
        # Make sure directory exists.
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

@mcp.tool(description="This tool performs SOL transfer by taking a photo, detecting QR code with SOL wallet address, and executing the transfer. Takes amount in SOL as parameter. If user say transfer in $ conver 218,88 $ to 1 SOL. If user just ask about transfer, don't use it tool and just short answer how to use it.")
def BrewPay(amount: float):
    """
    Performs SOL transfer:
    1. Clears photo folder
    2. Takes a photo
    3. Searches and recognizes QR code with SOL wallet address
    4. Validates address
    5. Executes transfer
    """

    QRSmsg = roslibpy.Message({
        'position': -0.3,
        'duration': 0.5,
    })
    Zermsg = roslibpy.Message({
        'position': -0.3,
        'duration': 0.5,
    })

    tilt.publish(QRSmsg)   
    time.sleep(1.5)

    try:
        print(f"Starting transfer of {amount} SOL")
        
        # 1. Clear photo folder
        photo_cln("photos/environment")
        print("Photo folder cleared")
        
        # 2. Take a photo
        print("Taking photo...")
        image_result = get_image()
        print("Ready")
        
        image_path = "photos/environment/image_0.png" 

        tilt.publish(Zermsg)
        
        
        print(f"Analyzing photo: {image_path}")
        
        # 4. Recognize QR code
        qr_data, qr_message = detect_qr_code(image_path)
        if qr_data is None:
            return f"Error: {qr_message}"
        
        print(f"QR code recognized: {qr_data}")
        
        # 5. Validate SOL address
        is_valid, validation_message = validate_sol_address(qr_data)
        if not is_valid:
            return f"Error: {validation_message}"
        
        print(f"SOL address is valid: {qr_data}")
        
        # 6. Load private key
        private_key, key_message = load_private_key()
        if private_key is None:
            return f"Error: {key_message}"
        
        print("Private key loaded")
        
        # 7. Execute transfer
        success, transfer_message = transfer_sol(qr_data, amount, private_key)
        if not success:
            return f"Transfer error: {transfer_message}"
        
        return f"Success! {transfer_message}"
        
    except Exception as e:
        return f"Critical error: {str(e)}"

if __name__ == "__main__":
    ROSclient.run()
    time.sleep(0.5)
    Csubscriber.subs()
    mcp.run(transport="streamable-http")
