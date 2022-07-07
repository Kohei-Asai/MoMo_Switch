setup:
	pip install ttkthemes
	pip install torch
	pip install pygame
	pip install bleak

make ble_setup:
	python3 arduino/ble_get_information.py

run:
	python3 main.py