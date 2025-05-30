from general_agent import AIAgent

def main():
    # Create agent instance
    try:
        agent = AIAgent()
        
        # Example queries
        queries = [
            "What is the current price of akamai technologies stock?",
            "What is the weather in each of the 5 biggest israel cities?",
            "Can you summarize the latest news on nvidia from the last 3 days?",
            "Please explain to me about langchain",
            "When was Elon Musk born and how old was he in 2022?"
        ]
        
        # Run example query (uncomment to test)
        result = agent.run(queries[0])
        print(result)
        
    except Exception as e:
        print(f"Error initializing agent: {e}")

if __name__ == "__main__":
    main()