import uvicorn

from spot_backend.config import get_settings


def main() -> None:
    s = get_settings()
    uvicorn.run("spot_backend.app:app", host=s.api_host, port=s.api_port, reload=False)


if __name__ == "__main__":
    main()
