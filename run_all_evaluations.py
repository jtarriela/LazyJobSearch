#!/usr/bin/env python3
"""
Run All Resume Evaluations Script

This script runs all three evaluation approaches:
1. Standard single resume evaluation (test_jdtarriela_anduril.py)
2. Enhanced single resume evaluation (enhanced_test_jdtarriela_anduril.py)  
3. Dual resume comparison evaluation (dual_resume_anduril_evaluation.py)

This provides a comprehensive view of both resumes against Anduril opportunities.
"""
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status"""
    try:
        console.print(f"\n🚀 [bold cyan]Running {description}...[/bold cyan]")
        console.print("=" * 80)
        
        result = subprocess.run([
            sys.executable, script_name
        ], cwd=Path(__file__).parent, capture_output=False, text=True)
        
        if result.returncode == 0:
            console.print(f"✅ [green]{description} completed successfully[/green]")
            return True
        else:
            console.print(f"❌ [red]{description} failed with exit code {result.returncode}[/red]")
            return False
            
    except Exception as e:
        console.print(f"❌ [red]Error running {description}: {e}[/red]")
        return False

def main():
    """Main function to run all evaluations"""
    
    console.print("[bold blue]🎯 LazyJobSearch: Complete Resume Evaluation Suite[/bold blue]")
    console.print()
    
    intro_panel = Panel(
        """This comprehensive evaluation will run:

📄 Individual Resume Evaluations:
• Standard evaluation of jtarriela_resume[sp].pdf
• Enhanced evaluation of jtarriela_resume[sp].pdf

📊 Comparative Analysis:
• Side-by-side comparison of both resume files
• Detailed scoring and recommendation analysis

🎯 Target: Anduril Industries Career Opportunities""",
        title="📋 Evaluation Overview",
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(intro_panel)
    console.print()
    
    # Track results
    results = []
    
    # 1. Run standard single resume evaluation
    success = run_script(
        "test_jdtarriela_anduril.py",
        "Standard Resume Evaluation (jtarriela_resume[sp].pdf)"
    )
    results.append(("Standard Evaluation", success))
    
    console.print("\n" + "─" * 80 + "\n")
    
    # 2. Run enhanced single resume evaluation  
    success = run_script(
        "enhanced_test_jdtarriela_anduril.py",
        "Enhanced Resume Evaluation (jtarriela_resume[sp].pdf)"
    )
    results.append(("Enhanced Evaluation", success))
    
    console.print("\n" + "─" * 80 + "\n")
    
    # 3. Run dual resume comparison
    success = run_script(
        "dual_resume_anduril_evaluation.py",
        "Dual Resume Comparison (Both Resumes)"
    )
    results.append(("Dual Resume Comparison", success))
    
    console.print("\n" + "=" * 80 + "\n")
    
    # Summary
    console.print("[bold green]🎉 All Evaluations Complete![/bold green]")
    console.print()
    
    # Results summary
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    summary_text = f"""📊 Execution Summary:

✅ Successful: {success_count}/{total_count} evaluations
"""
    
    for name, success in results:
        status = "✅ Success" if success else "❌ Failed"
        summary_text += f"• {name}: {status}\n"
    
    if success_count == total_count:
        summary_text += "\n🏆 All evaluations completed successfully!"
        summary_text += "\n\n💡 Review the outputs above to compare both resumes against Anduril opportunities."
        border_style = "green"
    else:
        summary_text += f"\n⚠️  {total_count - success_count} evaluation(s) had issues."
        border_style = "yellow"
    
    summary_panel = Panel(
        summary_text,
        title="📈 Final Results",
        border_style=border_style,
        padding=(1, 2)
    )
    
    console.print(summary_panel)
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())