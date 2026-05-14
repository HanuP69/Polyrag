"""
Create a simple test PDF for the PolyRAG pipeline.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.config import DATA_DIR

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed. Run: pip install PyMuPDF")
    sys.exit(1)


def create_test_pdf():
    """Create a multi-page test PDF with contract-like content."""
    
    pdf_path = os.path.join(DATA_DIR, "test_contract.pdf")
    doc = fitz.open()
    
    pages_content = [
        # Page 1: Introduction
        """SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of January 1, 2024, 
by and between TechCorp Inc. ("Provider") and ClientCo LLC ("Client").

WHEREAS, Provider is in the business of providing software development and 
consulting services; and WHEREAS, Client desires to engage Provider to 
perform certain services as described herein.

NOW, THEREFORE, in consideration of the mutual covenants and agreements 
set forth herein, the parties agree as follows:

1. SCOPE OF SERVICES

Provider shall perform the following services for Client:
(a) Design and development of a custom enterprise resource planning system
(b) Integration with existing Client databases and workflows
(c) Training and documentation for Client staff
(d) Ongoing maintenance and support for a period of 12 months""",

        # Page 2: Terms and Conditions
        """2. TERM AND TERMINATION

2.1 Term. This Agreement shall commence on the Effective Date and shall 
continue for a period of twenty-four (24) months, unless earlier terminated 
in accordance with this Section 2.

2.2 Termination for Convenience. Either party may terminate this Agreement 
upon ninety (90) days' prior written notice to the other party.

2.3 Termination for Cause. Either party may terminate this Agreement 
immediately upon written notice if the other party:
(a) Materially breaches this Agreement and fails to cure such breach 
    within thirty (30) days after receiving written notice thereof;
(b) Becomes insolvent or files for bankruptcy protection;
(c) Ceases to conduct business in the normal course.

2.4 Effect of Termination. Upon termination:
(a) Client shall pay Provider for all services performed through the 
    termination date;
(b) Provider shall deliver all work product completed to date;
(c) Each party shall return all confidential information of the other party.

3. COMPENSATION

3.1 Fees. Client shall pay Provider a monthly fee of $25,000 for the 
duration of this Agreement.

3.2 Expenses. Client shall reimburse Provider for all reasonable 
out-of-pocket expenses incurred in connection with the services.""",

        # Page 3: Liability and Indemnity
        """4. INDEMNIFICATION

4.1 Provider Indemnification. Provider shall indemnify, defend, and hold 
harmless Client and its officers, directors, employees, and agents from 
and against any and all claims, damages, losses, costs, and expenses 
(including reasonable attorneys' fees) arising out of or relating to:
(a) Provider's breach of this Agreement;
(b) Provider's negligent or willful misconduct;
(c) Any infringement of third-party intellectual property rights by 
    the work product delivered under this Agreement.

4.2 Client Indemnification. Client shall indemnify, defend, and hold 
harmless Provider from and against any claims arising from:
(a) Client's use of the work product in violation of this Agreement;
(b) Client's breach of applicable laws or regulations.

5. LIMITATION OF LIABILITY

5.1 IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, 
SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES ARISING OUT OF THIS AGREEMENT.

5.2 The total aggregate liability of Provider under this Agreement shall 
not exceed the total fees paid by Client during the twelve (12) month 
period preceding the event giving rise to the claim.

6. FORCE MAJEURE

Neither party shall be liable for any failure to perform its obligations 
under this Agreement if such failure results from circumstances beyond 
the reasonable control of that party, including but not limited to 
natural disasters, war, terrorism, pandemic, government actions, or 
failure of third-party infrastructure. The affected party shall provide 
prompt notice and use reasonable efforts to mitigate the impact.""",

        # Page 4: IP and Confidentiality
        """7. INTELLECTUAL PROPERTY

7.1 Work Product. All work product created by Provider under this 
Agreement shall be considered "work made for hire" and shall be the 
exclusive property of Client upon full payment.

7.2 Pre-existing IP. Provider retains all rights in its pre-existing 
intellectual property, tools, and methodologies. Provider grants Client 
a non-exclusive, perpetual license to use such pre-existing IP solely 
as incorporated in the work product.

8. CONFIDENTIALITY

8.1 Definition. "Confidential Information" means any non-public information 
disclosed by one party to the other, whether orally, in writing, or by 
inspection, including but not limited to trade secrets, business plans, 
technical data, customer lists, and financial information.

8.2 Obligations. The receiving party shall:
(a) Use Confidential Information solely for purposes of this Agreement;
(b) Not disclose Confidential Information to any third party without 
    prior written consent;
(c) Protect Confidential Information with the same degree of care used 
    for its own confidential information, but no less than reasonable care.

8.3 Duration. The obligations of confidentiality shall survive 
termination of this Agreement for a period of five (5) years.

9. GOVERNING LAW

This Agreement shall be governed by and construed in accordance with 
the laws of the State of Delaware, without regard to its conflicts 
of law principles."""
    ]
    
    for content in pages_content:
        page = doc.new_page(width=612, height=792)  # Letter size
        # Insert text with a readable font
        text_rect = fitz.Rect(72, 72, 540, 720)  # 1-inch margins
        page.insert_textbox(
            text_rect,
            content,
            fontsize=10,
            fontname="helv",
            align=0  # left align
        )
    
    doc.save(pdf_path)
    doc.close()
    print(f"Test PDF created: {pdf_path}")
    print(f"Pages: {len(pages_content)}")
    return pdf_path


if __name__ == "__main__":
    create_test_pdf()
