import pveagle

access_key = "eL6nbM9nK2bJGRm1I7mxY+t51zua+chhRFYJ1EgB5sYUBstEgYZcQg=="
eagle_profiler = pveagle.create_profiler(access_key)

def get_next_enroll_audio_data(num_samples):
    pass


percentage = 0.0
while percentage < 100.0:
    percentage, feedback = eagle_profiler.enroll(get_next_enroll_audio_data(eagle_profiler.min_enroll_samples))
    print(feedback.name)