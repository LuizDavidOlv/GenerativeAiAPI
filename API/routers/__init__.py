

from Routers import (OpenAiRouter, SpeechAndTextRouter, FineTunningRouter, AgentsRouter, HuggingFaceRouter, QuerySqlServerRouter, 
    JwtAuthenticationRouter, LangGraphRouter, EssayWriterRouter, LlamaIndexRouter) 


v1_routers = [AgentsRouter.router, EssayWriterRouter.router, FineTunningRouter.router, HuggingFaceRouter.router, 
              LangGraphRouter.router, OpenAiRouter.router, QuerySqlServerRouter.router, SpeechAndTextRouter.router, 
              JwtAuthenticationRouter.router, LlamaIndexRouter.router]

routers = {"v1:": v1_routers}