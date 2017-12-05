@ECHO OFF

if "%1" == "test" (
    py.test --cov calliope --cov-report term-missing
    goto end
)

if "%1" == "lint" (
    pylint calliope
    goto end
)

rem Note that `--python` removed from `mprof run` compared to BASH file counterpart
if "%1" == "profile" (
    set PROFILE=profile_%date:~-4%-%date:~3,2%-%date:~0,2%-%time:~0,2%-%time:~3,2%-%time:~6,2%
	mprof run -C -T 1.0 calliope run calliope/example_models/national_scale/model.yaml --override_file=calliope/example_models/national_scale/overrides.yaml:profiling --profile --profile_filename=%PROFILE%.profile
	pyprof2calltree -i %PROFILE%.profile -o %PROFILE%.calltree
	gprof2dot -f callgrind %PROFILE%.calltree | dot -Tsvg -o %PROFILE%.callgraph.svg
    goto end
)

if "%1" == "profile-clean" (
    del profile_*
	del mprofile_*
    goto end
)

:end
