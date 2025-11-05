#!/usr/bin/env python3
import uvicorn


if __name__ == "__main__":
    uvicorn.run("spot_scam.api.app:app", host="0.0.0.0", port=8000)
