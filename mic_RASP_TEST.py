# Установка необходимых библиотек:
# pip install pyaudio wave

import pyaudio
import wave
import os

def list_audio_devices():
    """
    Выводит список всех доступных аудиоустройств ввода (микрофонов)
    и возвращает их информацию.
    """
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    
    print("Доступные устройства ввода:")
    
    input_devices = []
    for i in range(device_count):
        device_info = p.get_device_info_by_index(i)
        # Проверяем, является ли устройство микрофоном (вводом)
        if device_info.get('maxInputChannels') > 0:
            input_devices.append((i, device_info.get('name')))
            print(f"Индекс: {i}, Имя: {device_info.get('name')}")
    
    p.terminate()
    return input_devices

def find_working_sample_rate(device_index):
    """
    Находит поддерживаемую частоту дискретизации для указанного устройства,
    тестируя стандартные значения.
    """
    p = pyaudio.PyAudio()
    # Список стандартных частот для тестирования
    test_rates = [48000, 44100, 22050, 16000, 11025, 8000]
    
    for rate in test_rates:
        try:
            # Пытаемся открыть и сразу закрыть поток, чтобы проверить, работает ли частота
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=rate,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=1024)
            stream.close()
            p.terminate()
            return rate
        except Exception:
            # Если произошла ошибка, это значит, что частота не поддерживается
            continue

    p.terminate()
    return None

def record_and_play_audio(device_index):
    """
    Записывает аудио с указанного микрофона, сохраняет его в файл и затем воспроизводит.
    """
    # Автоматически находим рабочую частоту для выбранного устройства
    RATE = find_working_sample_rate(device_index)
    if RATE is None:
        print("Не удалось найти подходящую частоту дискретизации для этого устройства. Попробуйте выбрать другое.")
        return

    # 1. Параметры записи
    CHUNK = 2048  # Увеличиваем размер буфера для стабильности
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RECORD_SECONDS = 5
    
    # 2. Определение пути для сохранения файла
    script_dir = os.path.dirname(os.path.abspath(__file__))
    WAVE_OUTPUT_FILENAME = os.path.join(script_dir, "output.wav")

    # 3. Инициализация PyAudio
    p = pyaudio.PyAudio()

    # 4. Запись аудио
    print(f"Начинаю запись с устройства с индексом {device_index} (Частота: {RATE} Гц)...")
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=device_index, 
                        frames_per_buffer=CHUNK)

        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("Запись завершена.")
        
        # 5. Остановка потока записи
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"Ошибка при записи: {e}")
        p.terminate()
        return

    # 6. Сохранение записанного аудио в файл
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print(f"Аудио сохранено в {WAVE_OUTPUT_FILENAME}")

    # 7. Воспроизведение аудио
    print("Начинаю воспроизведение...")
    try:
        with wave.open(WAVE_OUTPUT_FILENAME, 'rb') as wf:
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            
            data = wf.readframes(CHUNK)
            while data:
                stream.write(data)
                data = wf.readframes(CHUNK)
        print("Воспроизведение завершено.")
        
    except Exception as e:
        print(f"Ошибка при воспроизведении: {e}")
        
    finally:
        # 8. Завершение работы с PyAudio
        p.terminate()

if __name__ == "__main__":
    # Шаг 1: Вывести список устройств и попросить пользователя выбрать
    available_devices = list_audio_devices()
    
    if not available_devices:
        print("Не найдено доступных устройств ввода. Проверьте подключение микрофона.")
    else:
        while True:
            try:
                choice = int(input("Введите индекс микрофона, который хотите использовать: "))
                # Проверяем, существует ли выбранный индекс
                if choice in [dev[0] for dev in available_devices]:
                    record_and_play_audio(choice)
                    break
                else:
                    print("Неверный индекс. Пожалуйста, выберите индекс из списка.")
            except ValueError:
                print("Неверный ввод. Пожалуйста, введите число.")