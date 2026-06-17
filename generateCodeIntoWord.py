from pathlib import Path
from spire.doc import *

# Read Python file
code_string = Path("rl_phishing_demo.py").read_text(encoding="utf-8")

# Create a Word document
doc = Document()

# Add a section
section = doc.AddSection()
section.PageSetup.Margins.All = 60

# Add a paragraph
paragraph = section.AddParagraph()

# Insert code string to the paragraph
paragraph.AppendText(code_string)

# Create a paragraph style
style = ParagraphStyle(doc)
style.Name = "code"
style.CharacterFormat.FontName = "Consolas"
style.CharacterFormat.FontSize = 12
style.ParagraphFormat.LineSpacing = 12
doc.Styles.Add(style)

# Apply the style to the paragraph
paragraph.ApplyStyle("code")

# Save the document
doc.SaveToFile("codeInWord.docx", FileFormat.Docx2019)
doc.Dispose()
