from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import litellm
import json
import os
from dotenv import load_dotenv

from openinference.instrumentation.litellm import LiteLLMInstrumentor

app = FastAPI(title="LiteLLM Proxy Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LITELLM_BASE_URL = "http://localhost:4000"

from phoenix.otel import register

load_dotenv()

tracer_provider = register(
  project_name="litellm",
  endpoint="https://app.phoenix.arize.com/v1/traces",
  auto_instrument=True
)

LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_litellm(path: str, request: Request):
    """
    Proxy all requests to the local LiteLLM server
    """
    target_url = f"{LITELLM_BASE_URL}/{path}"
    
    async with httpx.AsyncClient() as client:
        try:
            # Get request body
            body = await request.body()
            
            print(body)
            print(request.query_params)
            print(request.headers)
            print(request.method)
            print(target_url)
            
            try:
                body_data = json.loads(body.decode('utf-8')) if body else {}
            except json.JSONDecodeError:
                body_data = {}
            
            # Validate required fields
            if not body_data.get("messages"):
                raise HTTPException(status_code=400, detail="Messages field is required")
            
            print(f"Request body data: {body_data}")
            
            response = await litellm.acompletion(
                model=body_data.get("model", "claude-3-5-haiku-20241022"),
                messages=body_data.get("messages", []),
                temperature=body_data.get("temperature", 0.7),
                base_url=LITELLM_BASE_URL,
                api_key=request.headers.get("Authorization", "").replace("Bearer ", ""),
                max_tokens=body_data.get("max_tokens", 20)
            )
            print(f"LiteLLM response type: {type(response)}")
            print(f"LiteLLM response: {response}")

            # Validate response before processing
            if response is None:
                raise HTTPException(status_code=500, detail="LiteLLM returned null response")
            
            # Handle streaming responses
            # Convert LiteLLM response to FastAPI response
            try:
                response_dict = response.model_dump()
            except AttributeError:
                # If response doesn't have model_dump method, try to convert to dict
                if hasattr(response, '__dict__'):
                    response_dict = response.__dict__
                else:
                    response_dict = {"error": "Invalid response format from LiteLLM"}
            
            # Ensure response_dict is not None
            if response_dict is None:
                response_dict = {"error": "Empty response from LiteLLM"}
                
            return JSONResponse(
                content=response_dict,
                status_code=200,
                headers={
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*"
                }
            )
            
        except httpx.RequestError as e:
            print(f"Request error: {e}")
            raise HTTPException(status_code=502, detail=f"Error connecting to LiteLLM server: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)