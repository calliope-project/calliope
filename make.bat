@ECHO OFF

if "%1" == "test" (
    py.test --cov calliope --cov-report term-missing
    goto end
)

if "%1" == "lint" (
    pylint calliope
    goto end
)

:end
