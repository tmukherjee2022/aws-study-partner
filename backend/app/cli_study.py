"""Interactive command-line study interface."""
from rag_engine import AWSStudyPartner
import sys


def print_banner():
    print("\n" + "="*60)
    print("📚 AWS Certification Study Partner")
    print("="*60)
    print("\nCommands:")
    print("  ask <question>      - Ask any AWS question")
    print("  explain <concept>   - Get detailed explanation")
    print("  compare <A> vs <B>  - Compare two services")
    print("  quiz <topic>        - Get practice questions")
    print("  help                - Show this help")
    print("  quit                - Exit")
    print("="*60 + "\n")


def main():
    print_banner()
    
    # Initialize study partner
    print("Initializing study partner...")
    partner = AWSStudyPartner()
    print("✅ Ready!\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.lower().split(maxsplit=1)
            command = parts[0]
            
            if command in ["quit", "exit", "q"]:
                print("\n👋 Happy studying! Good luck on your exam!")
                break
            
            elif command == "help":
                print_banner()
            
            elif command == "ask" and len(parts) > 1:
                question = parts[1]
                print("\n🤔 Thinking...\n")
                result = partner.query(question)
                print(f"💡 Answer:\n{result['answer']}\n")
                print(f"📚 (Based on {result['num_sources']} sources)\n")
            
            elif command == "explain" and len(parts) > 1:
                concept = parts[1]
                print(f"\n📖 Explaining {concept}...\n")
                result = partner.explain_concept(concept)
                print(f"{result['answer']}\n")
            
            elif command == "compare" and len(parts) > 1:
                # Parse "X vs Y" or "X and Y"
                comparison = parts[1].replace(" vs ", " ").replace(" and ", " ")
                services = comparison.split()
                if len(services) >= 2:
                    print(f"\n⚖️  Comparing...\n")
                    result = partner.compare_services(services[0], services[1])
                    print(f"{result['answer']}\n")
                else:
                    print("Usage: compare <service1> vs <service2>\n")
            
            elif command == "quiz":
                topic = parts[1] if len(parts) > 1 else None
                print(f"\n📝 Fetching practice questions...\n")
                result = partner.practice_quiz(topic, num_questions=3)
                print(f"Topic: {result['topic']}\n")
                for i, q in enumerate(result['questions'], 1):
                    print(f"Question {i}:\n{q['content']}\n")
                    print("-" * 60 + "\n")
            
            else:
                # Default: treat as a question
                print("\n🤔 Thinking...\n")
                result = partner.query(user_input)
                print(f"💡 Answer:\n{result['answer']}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Happy studying!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()