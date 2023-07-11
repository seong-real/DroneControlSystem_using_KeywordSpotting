from __future__ import print_function
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil  # Needed for command message definitions
import time
import math
import argparse
import pyaudio
import wave
import numpy as np
import tensorflow as tf 
import pathlib

parser = argparse.ArgumentParser(description='Commands vehicle using vehicle.simple_goto.')
parser.add_argument('--connect',
                    help="Vehicle connection target string. If not specified, SITL automatically started and used.")
args = parser.parse_args()

connection_string = args.connect
sitl = None

final_result = 0

# Start SITL if no connection string specified
if not connection_string:
    import dronekit_sitl

    sitl = dronekit_sitl.start_default()
    connection_string = sitl.connection_string()

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True)


def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    """ while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)"""

    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)  # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:  # Trigger just below target alt.
            print("Reached target altitude")
            break
        time.sleep(1)


def ascend(vehicle, target_altitude):
    current_altitude = vehicle.location.global_relative_frame.alt

    target_altitude = current_altitude + target_altitude
    vehicle.mode = VehicleMode("GUIDED")
    target_location = LocationGlobalRelative(
        vehicle.location.global_relative_frame.lat,
        vehicle.location.global_relative_frame.lon,
        target_altitude)
    vehicle.simple_goto(target_location)

    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        if current_altitude >= target_altitude * 0.95:
            print("목표 고도 도달")
            break
        time.sleep(2)


def decend(vehicle, target_altitude):
    current_altitude = vehicle.location.global_relative_frame.alt
    target_altitude = current_altitude - target_altitude
    vehicle.mode = VehicleMode("GUIDED")
    target_location = LocationGlobalRelative(
        vehicle.location.global_relative_frame.lat,
        vehicle.location.global_relative_frame.lon,
        target_altitude)
    vehicle.simple_goto(target_location)
    while True:
        current_altitude = vehicle.location.global_relative_frame.alt
        if current_altitude >= target_altitude * 0.90:
            print("목표 고도 도달")
            break
        time.sleep(2)


def condition_yaw(heading, relative=False):
    """
    Send MAV_CMD_CONDITION_YAW message to point vehicle at a specified heading (in degrees).
    """
    if relative:
        is_relative = 1  # yaw relative to direction of travel
    else:
        is_relative = 0  # yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,  # command
        0,  # confirmation
        heading,  # param 1, yaw in degrees
        0,  # param 2, yaw speed deg/s
        1,  # param 3, direction -1 ccw, 1 cw
        is_relative,  # param 4, relative offset 1, absolute angle 0
        0, 0, 0)  # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)


def get_location_metres(original_location, dNorth, dEast):
    earth_radius = 6378137.0  # Radius of "spherical" earth
    # Coordinate offsets in radians
    dLat = dNorth / earth_radius
    dLon = dEast / (earth_radius * math.cos(math.pi * original_location.lat / 180))

    # New position in decimal degrees
    newlat = original_location.lat + (dLat * 180 / math.pi)
    newlon = original_location.lon + (dLon * 180 / math.pi)
    if type(original_location) is LocationGlobal:
        targetlocation = LocationGlobal(newlat, newlon, original_location.alt)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation = LocationGlobalRelative(newlat, newlon, original_location.alt)
    else:
        raise Exception("Invalid Location object passed")

    return targetlocation;


def get_distance_metres(aLocation1, aLocation2):
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5


def goto_position_target_local_ned(north, east, down):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
        0b0000111111111000,  # type_mask (only positions enabled)
        north, east, down,  # x, y, z positions (or North, East, Down in the MAV_FRAME_BODY_NED frame
        0, 0, 0,  # x, y, z velocity in m/s  (not used)
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    # send command to vehicle
    vehicle.send_mavlink(msg)


def goto(dNorth, dEast, gotoFunction=vehicle.simple_goto):
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, dNorth, dEast)
    targetDistance = get_distance_metres(currentLocation, targetLocation)
    gotoFunction(targetLocation)

    while vehicle.mode.name == "GUIDED":
        remainingDistance = get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
        print("Distance to target: ", remainingDistance)
        if remainingDistance <= targetDistance * 0.1:
            print("Reached target")
            break;
        time.sleep(2)




DATASET_PATH = 'speech_commands_v0.02'
data_dir = pathlib.Path(DATASET_PATH)

# Define the input shape for the CNN model
input_shape = (124, 129, 1)

# Load the CNN model
model = tf.keras.models.load_model('test_spec.h5')
# Define the PyAudio settings
CHUNK_SIZE = 1024
SAMPLE_RATE = 16000
RECORD_SECONDS = 2

data_path = './output.wav'


def record_sound():
    # 설정값
    CHUNK = 1024  # 음성을 처리할 작은 조각(chunk) 크기
    FORMAT = pyaudio.paInt16  # pyaudio에서 사용할 bit depth
    CHANNELS = 1  # 모노
    RATE = 16000  # 샘플링 레이트 (Hz)
    RECORD_SECONDS = 1  # 음성을 몇 초간 녹음할지 지정
    WAVE_OUTPUT_FILENAME = "output.wav"  # 저장될 파일 이름

    # pyaudio 객체 생성
    p = pyaudio.PyAudio()

    # 스트림 객체 생성
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("3초전")
    time.sleep(1)
    print("2초전")
    time.sleep(1)
    print("1초전")
    time.sleep(1)
    print("* 녹음 시작")

    frames = []

    # 음성 녹음
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* 녹음 종료")

    # 스트림 정리, 객체 삭제
    stream.stop_stream()
    stream.close()
    p.terminate()

    # 음성 데이터를 wave 파일로 저장
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def get_spectrogram(filename):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        filename, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def model_input(data_path):
    x = tf.io.read_file(str(data_path))
    x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000, )
    x = tf.squeeze(x, axis=-1)
    waveform = x
    x = get_spectrogram(x)
    x = x[tf.newaxis, ...]

    prediction = model(x)

    # Use the model to predict the class of the audio signal
    result = model.predict(x)

    if np.argmax(result) == 0:
        print("예측은 backward 입니다")
    elif np.argmax(result) == 1:
        print("예측은 down 입니다")
    elif np.argmax(result) == 2:
        print("예측은 go 입니다")
    elif np.argmax(result) == 3:
        print("예측은 left 입니다")
    elif np.argmax(result) == 4:
        print("예측은 right 입니다")
    elif np.argmax(result) == 5:
        print("예측은 stop 입니다")
    elif np.argmax(result) == 6:
        print("예측은 up 입니다")

    return np.argmax(result)



# Arm and take of to altitude of 1 meters
arm_and_takeoff(20)

print("Set groundspeed to 1m/s.")
vehicle.groundspeed = 1

while (1):
    print("1. 음성녹음")
    print("2. 음성분석")
    print("3. 음성녹음&분석")
    print("4. RTL")
    state = int(input("기능을 선택하세요 :"))
    if state == 1:
        record_sound()
    elif state == 2:
        final_result = model_input(data_path)
    elif state == 3:
        record_sound()
        final_result = model_input(data_path)
    elif state == 4:
        print("Setting LAND mode...")
        vehicle.mode = VehicleMode("RTL")
        break

    order = input("명령을 실행하시겠습니까? Y/N")

    if (order == "Y"):
        if (final_result == 2):  # forward
            goto(5, 0)

        elif (final_result == 0):  # backward
            goto(-5, 0)

        elif (final_result == 3):  # left
            goto(0, -5)

        elif (final_result == 4):  # right
            goto(0, 5)

        elif (final_result == 6):  # up
            ascend(vehicle, 5)

        elif (final_result == 1):  # down
            decend(vehicle, 5)

        elif (final_result == 5):  # hovering
            goto(0, 0)
