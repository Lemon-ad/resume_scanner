import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer #converts documents into numerical word weights.
from sklearn.metrics.pairwise import cosine_similarity

st.title("Applicant Tracking System")
st.write("Upload resume and compare them with a job description")
job_description = st.text_area(
    "Job Description", 
    placeholder="Paste the job description here..."
)

resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Scan Resumes"):
    if not resume_file: 
        st.warning("Please Upload at least one resume")

    elif not job_description.strip(): 
        #.strip() removes spaces from the beginning and end. This lets us treat someone entering only spaces as an empty description.
        st.warning("Please provide a job description to compare resumes against")

    else:
        extracted_resume = []
        for files in resume_file:
            # Process each resume file
            st.write(f"Processing {files.name}...")
            try: 
                reader = PdfReader(files)
                resume_text = ""
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    resume_text += page_text + "\n"
                if not resume_text.strip():
                    st.warning(f"No text found in {files.name}. Please check the file format.")
                    continue
                extracted_resume.append({
                    "file_name": files.name,
                    "text": resume_text
                })
                with st.expander(f"View Text of {files.name}"):
                    st.text(resume_text)
                    
                st.success(f"Finished processing {files.name}")
            except Exception as error:
                st.error(f"Could not read {files.name}: {error}")

        if extracted_resume:
            documents = [job_description] + [resume["text"] for resume in extracted_resume]

            vectorizer = TfidfVectorizer(stop_words='english') #converts documents into numbers based on the words they contain, ignore a built-in list of common English words, such as “the” and “and.”

            try:
                tfidf_matrix = vectorizer.fit_transform(documents)
                similarity_scores = cosine_similarity(
                    tfidf_matrix[0:1], 
                    tfidf_matrix[1:]
                    ).flatten()
                st.subheader("Resume Similarity Scores")

                for resume, score in zip(extracted_resume, similarity_scores):
                    st.write(f"{resume['file_name']}: "
                             f"{score * 100:.1f}% text similarity")

            except ValueError as error:
                st.error(f"Could not convert text: {error}")
