from setuptools import setup, find_packages

setup(
    name="Dolphi_MediQ",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "langchain",
        "faiss-cpu",
        "nomic",
        "requests",
        "numpy",
        "pydantic",
        "python-docx",
        "PyMuPDF",        
    ],
    include_package_data=True,
    python_requires=">=3.10",
)
