from graders import grade_medium


def grader(response_text):
    # deterministic rubric for medium task
    required = ["sorry", "order", "update"]
    return grade_medium(response_text, required)
