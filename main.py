from src.tools.lab_tool import lab_test_suggester

result = lab_test_suggester.run(
    symptoms="fever",
    severity="",
    top_k=3,
)

print(result)