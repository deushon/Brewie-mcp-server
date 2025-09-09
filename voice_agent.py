import asyncio
import os
import json
import tempfile
import time
import speech_recognition as sr
from gtts import gTTS
import pygame
from fastmcp import Client
from together import Together
import pyaudio
import struct
import threading
from hashlib import md5
import roslibpy
import subprocess
import re

import io

from pvrecorder import PvRecorder

# ==============================
# Testing replacement of activation method
# ==============================
import pvporcupine

# Your Picovoice API key

ACCESSW_KEY = os.getenv("WAKEUP_API_KEY")


regcommand = [
    'eagle_demo_mic',
    'enroll',
    '--access_key',
    ACCESSW_KEY,
    '--output_profile_path',
    'master_sh/master_voice'
]




MCP_HOST="http://127.0.0.1:8000/mcp"

MCP_http_client = Client(MCP_HOST)

# Paths to your .ppn files (wake words)
KEYWORD_PATHS = [
    'robot_en_raspberry-pi_v3_0_0.ppn'
]
# ==============================
# Configuration
# ==============================

WAKE_WORD = "nex"
#MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
# ==============================
# TTS function
# ==============================

# Cache for audio files: {hash(text) -> file_path}
tts_cache = {}
# Lock for safe access to cache from different threads
cache_lock = threading.Lock()

# Flag for completion of work
tts_active = True

message_history = []
history_active = False
text_input = False
 
def get_files_in_directory(directory_path):

  files = []
  for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)
    if os.path.isfile(item_path):
      files.append(item)
  return files


def speak_with_gtts(text: str):
    """Speaks text using gTTS and pygame. Supports caching."""
    print(f"[TTS] Speech: {text}")
    filelist=get_files_in_directory("audio_out")
    
    if text in filelist:
        audio_file = "audio_out/"+text
        print(f"[TTS] Using cached audio for: {text}")
    else:
        # Create temporary file and save speech
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save("audio_out/"+text)
        audio_file = "audio_out/"+text
        print(f"[TTS] Generated new audio for: {text}")

    # Play in separate thread
    threading.Thread(target=play_audio, args=(audio_file,)).start()


def play_audio(file_path: str):
    """Plays audio file through pygame"""
    try:
        # Initialize mixer once
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy() and tts_active:
            time.sleep(0.1)

    except Exception as e:
        print(f"[TTS] Play error: {e}")


def stop_tts():
    """Stops playback and cleans up resources"""
    global tts_active
    tts_active = False
    pygame.mixer.quit()
    
    # Optional: remove temporary files from cache
    for path in tts_cache.values():
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[TTS] File deletion error: {e}")

# ==============================
# Speech recognition
# ==============================

def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("I'm listening...")
        audio = recognizer.listen(source)


    try:
        usersp = recognizer.recognize_google(audio, language="en-US").lower()
        wav_data = audio.get_wav_data()

        with open("audio_in/"+usersp, "wb") as f:
            f.write(wav_data)

        
        return usersp
    except sr.UnknownValueError:
        return None

# ==============================
# FastMCP client
# ==============================

async def call_mcp_tool(tool_name: str, parameters: dict):
    """Calls MCP server tool through FastMCP"""
    try:
        async with MCP_http_client:
            result = await MCP_http_client.call_tool(tool_name, parameters)
            print(f"[MCP] Result for {tool_name}: {result}")
            return True, result
    except Exception as e:
        print(f"[MCP] Error calling {tool_name}: {e}")
        return False, str(e)

# ==============================
# LLM + command processing
# ==============================

client = Together()
SYSTEM_PROMPT = None

async def init_system_prompt():
    """Form system prompt once"""
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is not None:
        return SYSTEM_PROMPT

    tools = []
    try:
        async with MCP_http_client:
            tools = await MCP_http_client.list_tools()
    except Exception as e:
        print(f"[LLM] Cannot get list of tools: {e}")
        tools = []

    result = await call_mcp_tool("get_available_actions", {})

    SYSTEM_PROMPT = (
        "You are a voice assistant controlling a robot through an MCP server.\n"
        "Available tools:\n" + 
        "\n".join([f"- {tool}" for tool in tools]) +
        "Available actions for tool 'run_action', for 'run_action': Use ONLY action names WITHOUT .d6a extension\n" +
        "\n".join([f"- {result}"]) + 
        "\n\nRespond ONLY in this JSON format:\n"
        "{\n"
        '  "answer": "Your response to the user",\n'
        '  "commands": [\n'
        '    {\n'
        '      "tool": "tool_name",\n'
        '      "params": {"param1": "value1"}\n'
        '    },\n'
        '    ...\n'
        '  ]\n'
        "}\n"
        "Rules:\n"
        "1. 'commands' must be a list (can be empty)\n"
        "2. Only include parameters if the tool requires them\n"
        "3. Keep your verbal response (answer) concise\n"
        "4. If user asks to perform actions, include them in commands\n"
        "5. For make_step use parametr x and z. x move robot left (1.0) and right (-1.0), z move robot forward(1.0) and back(-1.0) if you just move forward set z 1.0 and x 0 :\n"
        "Example for 'turn on the light':\n"
        "{\n"
        '  "answer": "Turning on the light",\n'
        '  "commands": [\n'
        '    {"tool": "light_on", "params": {}}\n'
        '  ]\n'
        "}\n"
    )
    
    return SYSTEM_PROMPT

async def handle_conversation(user_input: str):
    global message_history

    user_input = user_input.lower()
    print(f"[User]: {user_input}")

    system_prompt = await init_system_prompt()
    message_history = [
        {"role": "system", "content": system_prompt},
    ]

    message_history.append({"role": "user", "content": user_input})

    print(f"Waiting LLM...")
    try:
        response = None
        if history_active: 
            response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=message_history,
                    max_tokens=800,
                    temperature=0.2,
                    stop=["</s>"],
            )
        else:
            response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=800,
                    temperature=0.2,
                    stop=["</s>"],
            )

        answer = response.choices[0].message.content.strip()
        print(f"[LLM] Raw response:\n{answer}")

        # Improved JSON processing
        json_str = answer
        if "```json" in answer:
            json_str = answer.split("```json")[1].split("```")[0].strip()
        elif "```" in answer:
            json_str = answer.split("```")[1].strip()
        
        try:
            response_data = json.loads(json_str)
             
            # Process commands
            toolis = False    
            commands = response_data.get("commands", [])
            if not isinstance(commands, list):
                commands = [commands]
            master_talk = False
            actrig = False
            for command in commands:
                tool_name = command.get("tool")
                if (tool_name == "sniper" or tool_name =="defend" or tool_name == "BrewPay"):
                    actrig = True
                    toolis = True
                
            if actrig:
                speak_with_gtts("Check voice")
                checkcommand = [
                    'eagle_demo_file',
                    'test',
                    '--access_key',
                    ACCESSW_KEY,
                    '--input_profile_paths',
                    'master_sh/master_voice',
                    '--test_audio_path',
                    'audio_in/'+user_input
                ]
                egrez=subprocess.run(checkcommand, check=True, capture_output=True, text=True)

                lines = egrez.stdout.strip().split('\n')
                parsed_data = []

                # Regular expression to find time and score
                pattern = r"time: (\d+\.\d+) sec \| scores -> `master_voice`: (\d+\.\d+)"

                for line in lines:
                    match = re.search(pattern, line)
                    if match:
                        time = float(match.group(1))
                        score = float(match.group(2))
                        parsed_data.append({'time': time, 'score': score})


                scores = [item['score'] for item in parsed_data if item['score'] != 0]
                print (scores)
                if(len(scores)>0):
                    average_score = sum(scores) / len(scores)
                else:
                    average_score=0
                print("The authenticity of the master's identity:")
                print(average_score)

                if(average_score>0.1):
                    master_talk = True

            if (master_talk):
                speak_with_gtts("Master verified")
                
            errAc = False
            for command in commands:
                if not isinstance(command, dict):
                    continue
                
                

                permissions = False
                params = command.get("params", {})

                tool_name = command.get("tool")
                if ((tool_name == "sniper" or tool_name =="defend" or tool_name == "BrewPay") and master_talk) or (tool_name != "sniper" and tool_name !=  "defend" and tool_name != "BrewPay"):
                    permissions = True
                    
                print(permissions)
                if tool_name:
                    if permissions:
                        success, result = await call_mcp_tool(tool_name, params )
                        if not success:
                            speak_with_gtts(f"Failed to execute {tool_name}")
                        else:
                            # Handle MCP result format - it's a list of TextContent objects
                            if isinstance(result, list) and len(result) > 0:
                                # Extract text from TextContent object
                                text_content = result[0]
                                if hasattr(text_content, 'text'):
                                    speak_with_gtts(text_content.text)
                                else:
                                    speak_with_gtts("Command executed successfully")
                            elif isinstance(result, dict) and "text" in result:
                                speak_with_gtts(result["text"])
                            elif isinstance(result, str):
                                speak_with_gtts(result)
                            else:
                                speak_with_gtts("Command executed successfully")
                    else:
                        speak_with_gtts("You do not have permission to perform this action.")
                        errAc=True
                else:
                    print("[LLM] Missing tool name in command")
                    

            # Add assistant response to history
            message_history.append({"role": "assistant", "content": answer})

            # Speak the response
            verbal_response = response_data.get("answer", "I'll execute your request")
            if not errAc and not toolis:
                speak_with_gtts(verbal_response)


        except json.JSONDecodeError as e:
            print(f"[LLM] JSON decode error: {e}")
            speak_with_gtts("I had trouble processing your request")
        except Exception as e:
            print(f"[LLM] Response handling error: {e}")
            speak_with_gtts("Something went wrong with my response")

    except Exception as e:
        print(f"[LLM] Request error: {e}")
        speak_with_gtts("I couldn't process your request")

# ==============================
# Main loop
# ==============================



async def main():
    

    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    time.sleep(1)

    pan = roslibpy.Topic(client, '/head_pan_controller/command', 'std_msgs/Float64')
    tilt = roslibpy.Topic(client, '/head_tilt_controller/command', 'std_msgs/Float64')
    action = roslibpy.Topic(client, '/app/set_action', 'std_msgs/String')

    headUPmsg = roslibpy.Message({
        'position': 0.2,
        'duration': 0.5,
    })

    headDOWNmsg = roslibpy.Message(    {
        'position': 0.0,
        'duration': 0.5,
    })

    panFXmsg = roslibpy.Message(    {
        'position': 0.0,
        'duration': 0.5,
    })

    standBmsg = roslibpy.Message(    {
        'data': 'walk_ready'
    })

    thinkmsg = roslibpy.Message(    {
        'data': 'think'
    })

    s1msg = roslibpy.Message({
        'position': 1,
        'duration': 0.25,
    })
    s2msg = roslibpy.Message({
        'position': -1.2,
        'duration': 4.5,
    })
    s3sg = roslibpy.Message({
        'position': 0,
        'duration': 0.5,
    })


    recognizer = sr.Recognizer()
    mic = sr.Microphone(sample_rate=16000)

    pan.publish(panFXmsg)


    filelist=get_files_in_directory("master_sh")

    tilt.publish(headUPmsg)
    
    if "master_voice" not in filelist:     
        speak_with_gtts("Welcome new master ")
        time.sleep(1.6)
        speak_with_gtts("Tell me a little about yourself so I can remember your voice.")
        time.sleep(4.2)
        #TODO Error handling
        subprocess.run(regcommand, check=True, capture_output=True, text=True)
        speak_with_gtts("New master registered!")
        time.sleep(2)
    
    
        
    
    speak_with_gtts("I am ready")
    time.sleep(1)
    tilt.publish(headDOWNmsg)



    
    
    #TODO
    #Manual microphone passthrough doesn't work, frequency error
    #Works only when starting with "default".
    #For this, run once with external audio disabled
    #Connect the needed microphone
    #Then the built-in speaker can be returned
    #Maybe just boot without speaker or unplug it to change device index?
    # Create Porcupine instance
    porcupine = pvporcupine.create(
        access_key=ACCESSW_KEY,
        keyword_paths=KEYWORD_PATHS,
        sensitivities= [0.9]* len(KEYWORD_PATHS)
    )

    # Audio stream setup
    recorder = PvRecorder(
        frame_length=porcupine.frame_length,
        #device_index=1
        )
    recorder.start()

    #Dictionary related to shooting
    search_words = ["save", "safe", "attack", "enemy", "opponent","fire","snipe","sniper"]
    print(f"Say Robot to start...")
    try:
        while True:

            if text_input:
                user_query = input(str(""))
                if user_query == "p":
                    await call_mcp_tool("get_image", "")
                if user_query == "s":
                    scom = {'targediscr': 'purle ball'}
                    await call_mcp_tool("sniper", scom)
                if user_query == "t":
                    scom = {'amount': 0.0001}
                    await call_mcp_tool("BrewPay", scom)
            
            pcm = recorder.read()

            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                print("Wake word detected!")
                tilt.publish(headUPmsg)
                #speak_with_gtts("Yes?")
                print("Listening for command...")
                user_query = recognize_speech_from_mic(recognizer, mic)
                if user_query:       
                    
                    tilt.publish(headDOWNmsg)
                    if any(word in user_query for word in search_words):
                        speak_with_gtts("I'll scout out the situation....")
                        pan.publish(s1msg)
                        time.sleep(0.5)
                        pan.publish(s2msg)
                    #else:
                        #action.publish(thinkmsg)
                        #speak_with_gtts("Thinking...")
                    await handle_conversation(user_query)
                    pan.publish(panFXmsg)
                    tilt.publish(panFXmsg)

                else:
                    speak_with_gtts("I didn't catch that")
                    tilt.publish(headDOWNmsg)
                    pan.publish(panFXmsg)

                    
    except KeyboardInterrupt:
        print("\n User stop")

    finally:
        # Clean up resources
        print("\n Cleaning...")
        action.unadvertise()
        tilt.unadvertise()
        pan.unadvertise()
        client.terminate()
        stop_tts()
        if 'porcupine' in locals():
            porcupine.delete()


if __name__ == "__main__":
    asyncio.run(main())