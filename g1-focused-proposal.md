# COS30018 Intelligent Systems - Option D Project Proposal

---

**To:** Project Supervisor Khoa Pham
**From:** He Thong Vo Luong Khong Xu - Group of 5 Students
**Date:** August 12, 2025
**Subject:** Project Proposal - LLM-Powered Business Improvement Agent (G1 Focus)

---

## Introduction

We are pleased to submit our formal project proposal for COS30018 – Intelligent Systems, Project Assignment Option D, focusing on Large Language Model (LLM)-powered Autonomous Agents and Multiagent Systems. Our team of five second and third-year university students has elected to pursue a **primary focus on Goal 1 (G1): Develop and deploy an LLM-powered autonomous agent**.

**Primary Objective (G1)**: We have selected the **Business Improvement Agent** mission as our core deliverable. Our proposed agent will analyze customer review datasets to provide comprehensive sentiment analysis, identify key thematic patterns, and generate actionable recommendations for business improvement.

**Secondary Objective (G2)**: If time and resources permit after successfully completing G1, we will explore extending our single agent into a multiagent system with specialized sub-agents for different aspects of business analysis.

Our strategic approach prioritizes delivering a robust, fully-functional autonomous agent that demonstrates sophisticated LLM-powered capabilities rather than attempting to build multiple less-developed agents.

## Project Strategy & Scope

### Primary Focus: Single Agent Excellence (G1)

Our main effort will concentrate on creating a sophisticated, autonomous Business Improvement Agent that can:

- **Autonomously process** large review datasets without human intervention
- **Intelligently decide** which analysis tools to use based on data characteristics
- **Generate comprehensive insights** with minimal user input
- **Provide actionable recommendations** with business impact prioritization
- **Learn and adapt** from different types of review data

### Secondary Goal: Multiagent Extension (G2) - Time Permitting

Only after completing G1, we may extend our system to include:

- **Specialist Agents**: Sentiment Specialist, Topic Analysis Specialist, Insight Generator
- **Coordinator Agent**: Orchestrates the specialist agents
- **Quality Control Agent**: Validates and cross-checks results

## Proposed System Architecture (G1 Focus)

Our Business Improvement Agent will employ a sophisticated, single-agent architecture with six integrated components:

### 1. Dataset Integration Module

Handles comprehensive dataset management:

- **Multi-Format Support**: CSV, JSON, Parquet, Excel
- **Dataset Discovery**: Automatic schema detection and validation
- **Intelligent Sampling**: Dynamic sampling strategies for large datasets
- **Data Quality Assessment**: Automated quality scoring and filtering

### 2. Advanced Data Preprocessing Module

Sophisticated data preparation capabilities:

- **Intelligent Text Cleaning**: Context-aware normalization and standardization
- **Language Processing**: Multi-language detection and translation support
- **Feature Engineering**: Automatic extraction of metadata features (review length, rating correlation, temporal patterns)
- **Bias Detection**: Identification and mitigation of dataset biases



### 3. Indexer - Embeddings and Vector DB

Convert review text to vector embeddings and store them to enable fast retrieval. 

- **Embedding library**: sentence-transformers, all-MiniLM-L6-v2
- **Local store**: FAISS, use IndexFlatIP with L2-normalized vectors to get cosine similarity


### 4. Core Autonomous Agent (The "Brain")

The sophisticated central intelligence component:

- **Agent Framework**: use LLM for high-level reasoning
- **LLM Support**: Google AI Studio
- **Dynamic Tool Selection**: Intelligent choice of analysis tools based on data characteristics
- **Self-Reflection**: Every claim must cite evidence
- **Memory Management**: Persistent context and learning from previous analyses
- **Auditable**: log tool call, prompts, evidence ids in final JSON output for evaluation

### 5. Comprehensive Analysis Toolkit (Agent's Arsenal)

Advanced analysis capabilities at the agent's disposal:


#### Retriever
Use FAISS local index + vector DB

#### Advanced Sentiment Analysis Tool

- **MVP implementation**: Sentiment analysis using Hugging Face
- **Stretch**: multi-dimensional emotion, fine-grained analysis of specific product/service features


#### Intelligent Topic Modeling Tool

- **MVP**: topic discovery using sentence-transformers and HDBSCAN

#### Contextual Summarization Tool

- **Multi-Level Summaries**: Executive, detailed, and technical summary levels
- **Pattern**: Input aggregated stats and evidence snippets, output summary with evidence

#### Strategic Insight Generation Tool

- **Heuristic scoring formula**
- **Business Impact Scoring**: Quantified priority rankings for recommendations
- **Root Cause Analysis**: Deep investigation of identified issues
- **Recommendation**: Include estimated effort, priority, evidence

### 6. Interactive User Interface

Comprehensive web-based interface:

- **Agent Chat Interface**: Natural language interaction with the autonomous agent
- **Dataset Management Dashboard**: Visual dataset exploration and selection
- **Real-Time Analysis Monitoring**: Live view of agent decision-meaking process if possible
- **Visualization Suite**: Advanced charts, graphs, and business intelligence displays

### 7. Comprehensive Evaluation Module

Rigorous testing and validation:

- **Multi-Metric Evaluation**: Accuracy, relevance, actionability of insights
- **Benchmark Comparisons**: Performance against established datasets
- **Performance Analytics**: Speed, resource usage, and scalability metrics

## Proposed Technology Stack

### Core Agent Development

- **Programming Language**: Python 3.9+
- **Agent Frameworks**: LangChain (primary), LlamaIndex
- **LLM Integration**: Gemini API, Hugging Face Transformers
- **Local LLM Serving**: Ollama, vLLM for local model deployment

### Large Language Models

- **Cloud Models**: Gemini Pro
- **Local Models**: Llama 3.1 (8B/70B), Mistral 7B, Qwen 2.5...
- **Specialized Models**: Fine-tuned sentiment analysis models

### Data Processing & Analysis

- **Data Manipulation**: Pandas, Polars for high-performance data processing
- **Text Processing**: spaCy, NLTK, transformers for NLP tasks
- **Machine Learning**: scikit-learn, torch for analysis components

### User Interface & Visualization

- **Web Framework**: Streamlit (primary), FastAPI + React (if needed)
- **Visualization**: Plotly, Altair for interactive visualizations
- **UI Components**: Streamlit-extras for enhanced interface elements

### Development Infrastructure

- **Version Control**: Git with GitHub for collaborative development
- **Documentation**: OneDrive + MkDocs for comprehensive documentation
- **Testing**: pytest, locust for unit and performance testing
- **Deployment**: Docker for containerization, Streamlit Cloud for hosting

## Success Metrics & Evaluation Criteria (G1 Focus)

### Primary G1 Success Metrics:

- **Autonomous Operation**: Agent completes analysis tasks with minimal human intervention
- **Analysis Quality**: Generated insights demonstrate clear business value and actionability
- **LLM Integration**: Seamless operation with both cloud and local LLM models

### Technical Performance Targets:

- **Sentiment Analysis Accuracy**: >90% accuracy on benchmark datasets
- **System Reliability**: 99.5% uptime during demonstration period
- **Response Quality**: User satisfaction >8/10 for generated recommendations

### G2 Extension Criteria (If Achieved):

- **Agent Coordination**: Successful coordination between specialized agents
- **Performance Improvement**: Multiagent system shows measurable improvement over single agent
- **Communication Efficiency**: Effective inter-agent communication and task distribution

## Risk Management & Contingency Planning

### G1 Primary Risks:

- **Agent Complexity**: Focus on core autonomous capabilities rather than advanced features
- **LLM Performance**: Maintain multiple LLM options for reliability
- **Integration Challenges**: Prioritize modular design for independent testing

### G2 Secondary Risks:

- **Time Constraints**: G2 is explicitly secondary - no compromise on G1 quality
- **Complexity Overhead**: Simple agent decomposition rather than complex multiagent architecture
- **Resource Allocation**: Maintain focus on G1 until completion

## Concluding Statement

Our team is committed to delivering a sophisticated, autonomous LLM-powered Business Improvement Agent that demonstrates advanced capabilities in independent analysis and decision-making. By prioritizing G1 completion, we ensure a robust, working deliverable that showcases the full potential of autonomous agents in business intelligence.

Our strategic focus on single-agent excellence will result in a more sophisticated and capable system than attempting to split resources between multiple less-developed agents. If time permits after G1 completion, we will extend toward G2 with a clear understanding of our system's capabilities and limitations.

We are confident in our ability to deliver a high-quality autonomous agent that meets academic requirements while demonstrating practical business applications. Our phased approach ensures continuous progress monitoring and adaptation as needed.

We look forward to beginning this focused development effort and welcome feedback on our G1-priority approach.

---

**Respectfully submitted by:**
He Thong Vo Luong Khong Xu
COS30018 Intelligent Systems
August 12, 2025
