from general_agent import AIAgent
# from db_agent import DBAgent  # noqa

def main():
    # Create agent instance
    try:
        agent = AIAgent()
        # agent = DBAgent(db_url="sqlite:///Chinook.db")
        
        # Example queries
        queries = [
            "What is the current price of akamai technologies stock?",
            "What is the weather in each one of the 5 biggest israel cities? (each one have different weather)  -give me the answer in structured format {'{{city}}': {desc, temperature, humidity, wind_speed, wind_direction, cloud_cover, precipitation, visibility, pressure, uv_index, sunrise, sunset}}",
            "Can you summarize the latest news on nvidia from the last 3 days?",
            "Please explain to me about langchain",
            "When was Elon Musk born and how old was he in 2022?"
        ]

        question = queries[1]
        # question = "Which sales agent made the most in sales in 2009?"

        # Run example query (uncomment to test)
        result = agent.run(question)
        print(result)
        
    except Exception as e:
        print(f"Error initializing agent: {e}")

if __name__ == "__main__":
    main()


    