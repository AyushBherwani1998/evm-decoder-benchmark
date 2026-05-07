import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")
LANGFUSE_PROJECT_ID = os.getenv("LANGFUSE_PROJECT_ID")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
RPC_URL = os.getenv("RPC_URL", "https://base.api.pocket.network")

BENCHMARK_MSG_VALUE_WEI = 250000000000000000
TO_ADDRESS = "0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24"
CALLDATA = "0xf305d719000000000000000000000000825b8f8d87b7b188c67753659f65b631148f44bf0000000000000000000000000000000000000000033b2e3c9fd0803ce80000000000000000000000000000000000000000000000033b2e3c9fd0803ce800000000000000000000000000000000000000000000000000000003782dace9d90000000000000000000000000000d80eead28a895cd00738f4820b6fd83881d9ed5c0000000000000000000000000000000000000000000000000000000069f9b219"

# ERC-20 Transfer
# BENCHMARK_GROUND_TRUTH = {
#     "action": "transfer",
#     "recipient": "0xA48a572D05a423EA2bB1B0E6049E7ED190A0CF83",
#     "amount": 10.2006,
#     "token": {
#         "symbol": "USDC",
#         "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
#     }
# }

# Swap 
# BENCHMARK_GROUND_TRUTH = {
#     "action": "swap",
#     "input_token": {"symbol": "ETH", "address": "None"},
#     "output_token": {
#         "symbol": "SURGE",
#         "address": "0xedB6970B7BE5A522cFf14c8b679a2D813FF97b4D",
#     },
#     "amountOutMin": 8712.4811,
#     "deadline": 1777877917,
#     "recipient": "0xE97a87a29fd55786154E7c808bE75E20A8060EC3",
# }


BENCHMARK_GROUND_TRUTH = {
    "action": "add_liquidity_eth",
    "token_a": {"symbol": "ETH", "address": None},
    "token_b": {
        "symbol": "POWERBALL",
        "ticker": "BALL",
        "address": "0x825b8F8D87B7b188c67753659F65B631148f44Bf",
    },
    "amount_eth": "0.25",
    "amount_token_desired": "1000000000",
    "amount_token_min": "1000000000",
    "amount_eth_min": "0.25",
    "deadline_utc": "1777971737",
    "recipient": "0xd80EEAD28A895CD00738f4820B6Fd83881d9eD5C",
}

ITERATIONS = 100
JUDGE_PROVIDER = "litellm"
JUDGE_MODEL = "claude-sonnet-4.6"
