import pathlib
import os
import streamlit.web.bootstrap as bootstrap

HERE = pathlib.Path(__file__).parent


flag_options = {
    "global.developmentMode": False,
    "server.headless": True,
    "server.port": os.getenv("PORT", 5000),
}

def app():
    bootstrap.load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True
    bootstrap.run(
        str(HERE.joinpath("app.py")),
        command_line="streamlit run",
        args=list(),
        flag_options=flag_options,
    )


if __name__ == "__main__":
    app()
