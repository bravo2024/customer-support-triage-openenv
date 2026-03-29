from graders import EasySample, grade_easy


def grader(predicted_labels, truth_labels):
    samples = [EasySample(p, t) for p, t in zip(predicted_labels, truth_labels)]
    return grade_easy(samples)
