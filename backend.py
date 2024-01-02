from gensim.summarization import summarize
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import os

# Check if the T5 model is already downloaded
model_name = "t5-small"
if not os.path.exists(model_name):
    # Download the T5 model
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
    t5_tokenizer = T5Tokenizer.from_pretrained(model_name)

# Example legal document
legal_document = """
    To All to Whom these presents shall come, I ABC, Son/Daughter of XYZ, aged 15, residing at Kolkata
Whereas I am desirous of appointing some fit and proper person to look after all my immovable properties, business and other affairs and requested <Name of Person Receiving Powers>, Son/Daughter of <Father’s Name>, aged <Age in Years>, residing at <Address> (hereinafter called ‘the Attorney’) to act for me and manage and look after my affairs which the Attorney has consented to do
NOW KNOW YOU ALL AND THESE PRESENTS WITNESS that I, the said and do hereby appoint the said Attorney as my true and lawful Attorney with full power and authority to do and execute all acts, deeds, and things as hereinafter mentioned.

To contract with any person for leasing for such period at such rent subject to such conditions as the attorney shall see fit, all or any of the said premises and any such person, to let into possession thereof and to accept surrenders of leases and for that purpose to make and execute any lease or grant or other lawful deed or instrument whatsoever which shall be necessary or proper in that behalf.
To pay or allow all taxes, rates, assessments, charges. deductions, expenses and all other payments and outgoings whatsoever due and payable or to become due and payable for or on account of my said lands, estates and premises.
To enter into and upon my lands and buildings and structures whatsoever and to view the state and defects for the reparation thereof and forthwith to give proper notices and directions for repairing the same and to let manage and Improve the same to the best advantage and to make or repair drains and roads thereon.
To sell (either by public auction or private treaty) or exchange and convey transfer and assign any of my lands and buildings and other property for such consideration and subject to such covenants as the Attorney may think fit and to give receipts for all or any part of the purchase or other consideration money And the same or any of them with like power, to mortgage charge or encumber and also to deal with my immovable personal property or any part thereof as the Attorney may think fit for the purpose of paying off reducing consolidating, or making substitution for any existing or future mortgage. charge, encumbrance. hypothecation or pledge of the same or any part thereof as the Attorney shall think fit and in general to sanction any scheme for dealing with mortgages, charges hypothecations or pledges of any property or any part thereof as fully and effectually as I myself could have done.
To purchase, take on lease or otherwise acquire such lands, houses, tenements and immovable property generally as the Attorney may think fit or desirable.
To enter into any development agreement with any developer or builder authorising him to develop any of my properties as mentioned above and to do and execute all acts and deeds as may be required to be done or executed.
To continue and or to open new, current and or overdraft accounts in my name with any Banks or Bankers and also to draw cheques and otherwise to operate upon any such accounts.
To engage, employ and dismiss any agents, clerks, servants or other persons in and about the performance of the purposes of these presents as the Attorney shall think fit.
To settle any account or reckoning whatsoever wherein I now am or at any time hereafter shall be in anywise interested or concerned with any person whomsoever and to pay or receive the balance thereof as the case may require.
To defend any suit or legal proceedings taken against me in any court of law and to do all acts and things as are mentioned above.

Signed, scaled and delivered in the presence of <Witness Details>
"""

# Extractive summarization
extractive_summary = summarize(legal_document, ratio=0.2)  # Adjust the ratio as needed

# Abstractive summarization using T5
summarizer = pipeline("summarization")
abstractive_summary = summarizer(
    legal_document,
    max_length=150,
    min_length=50,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True,
)

# Print the results
print("Extractive Summary:")
print(extractive_summary)

print("\nAbstractive Summary:")
print(abstractive_summary[0]["summary_text"])
