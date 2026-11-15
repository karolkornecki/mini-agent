from mini_agent.permissions.approval import request_approval


def test_approval_accepts_yes():
    assert request_approval(
        "write_file",
        {"path": "hello.txt"},
        ask=lambda _: "y",
    )


def test_approval_rejects_no():
    assert not request_approval(
        "write_file",
        {"path": "hello.txt"},
        ask=lambda _: "n",
    )


def test_approval_defaults_to_reject():
    assert not request_approval(
        "write_file",
        {"path": "hello.txt"},
        ask=lambda _: "",
    )
