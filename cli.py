from rag_engine import agent_pipeline, ingest_data
import time
import os
import sys
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_typing_effect(text, delay=0.02):
    """Print text with a typing effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def print_header():
    """Print stylized header."""
    header = """
    ╔════════════════════════════════════════════════╗
    ║                                                ║
    ║        RAG (Retrieval-Augmented Generation)    ║
    ║                Knowledge Assistant             ║
    ║                                                ║
    ╚════════════════════════════════════════════════╝
    """
    print(Fore.CYAN + header)

def print_divider():
    """Print a divider line."""
    print(Fore.BLUE + "="*60)

def print_section(title, content, color=Fore.GREEN):
    """Print a nicely formatted section."""
    print(color + f"\n▓▒░ {title} ░▒▓")
    print(Style.RESET_ALL + content)

def main():
    """
    Main function to run the CLI.
    It initializes the CLI and handles user input.
    It uses the agent_pipeline function to process the input query.
    """
    try:
        # Test Ollama connection first
        print(Fore.YELLOW + "Testing Ollama connection...")
        print_header()
        print(Fore.YELLOW + "Initializing knowledge base...")
        print(Fore.GREEN + "✓ Knowledge base loaded successfully!" + Style.RESET_ALL)
        
        while True:
            print_divider()
            query = input(Fore.CYAN + "\n🔍 Ask a question " + Fore.WHITE + "(or " + Fore.RED + "'exit'" + Fore.WHITE + " to quit): " + Style.RESET_ALL)
            
            if query.lower() == 'exit':
                print_typing_effect(Fore.YELLOW + "\nThank you for using the RAG Knowledge Assistant. Goodbye!")
                break
            
            print(Fore.YELLOW + "\nProcessing your query...\n")
            # Add a small loading animation
            for _ in range(3):
                for c in ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]:
                    sys.stdout.write(f"\r{Fore.YELLOW}Processing {c}")
                    sys.stdout.flush()
                    time.sleep(0.1)
            
            tool, docs, answer = agent_pipeline(query)
            print("\r" + " " * 20 + "\r", end="")  # Clear the loading animation
            
            print_section(f"🛠️  TOOL USED", tool, Fore.MAGENTA)
            
            if docs:
                print_section("📚 RETRIEVED CONTEXT", "", Fore.BLUE)
                for i, doc in enumerate(docs):
                    snippet = doc.page_content[:500]
                    if len(doc.page_content) > 500:
                        snippet += "..."
                    print(f"{Fore.CYAN}╔═ Document {i+1} ═╗")
                    print(f"{Style.RESET_ALL}{snippet}\n")
            
            print_section("💡 ANSWER", answer, Fore.GREEN)
            print("\n" + Fore.YELLOW + "Would you like to ask another question?" + Style.RESET_ALL)
    except ConnectionError as e:
        print(Fore.RED + f"Error connecting to Ollama: {str(e)}")
        print(Fore.YELLOW + "Please ensure Ollama is running and the llama3 model is installed.")
        return
    except Exception as e:
        print(Fore.RED + f"Error in initialization: {str(e)}")
        import traceback
        print(Fore.RED + traceback.format_exc())
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.RED + "\n\nOperation canceled by user. Exiting...")
    except Exception as e:
        print(Fore.RED + f"\n\nAn error occurred: {e}")