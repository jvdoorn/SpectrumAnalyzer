import hashlib

from spectral.data.signal import Signal

BUFFER_SIZE = 65536

TEST_SAMPLE_RATE = 20000
TEST_SAMPLES = 10000
TEST_FREQUENCY = 5
TEST_AMPLITUDE = 5
TEST_DF = 20
ACCEPTABLE_ERROR = 0.2

TEST_SIGNAL = Signal.generate(TEST_SAMPLE_RATE, TEST_SAMPLES, TEST_FREQUENCY, TEST_AMPLITUDE)


def calculate_file_hash(file):
    file_md5 = hashlib.md5()

    with open(file, 'rb') as f:
        while True:
            data = f.read(BUFFER_SIZE)
            if not data:
                break
            file_md5.update(data)

    return file_md5.hexdigest()


def equal_file_hash(master_file, target_file):
    master_md5 = calculate_file_hash(master_file)
    target_md5 = calculate_file_hash(target_file)
    return master_md5 == target_md5
