1. Context Only
CONTEXT:
{retrieved_documents}

QUESTION:
{user_question}

INSTRUCTIONS:
Answer the QUESTION using only the information provided in the CONTEXT above.
Keep your answer grounded in the facts of the CONTEXT.
Use [chunk_id] notation immediately after each statement to cite sources.
If the CONTEXT doesn't contain enough information to fully answer the QUESTION, 
state: "I don't have enough information to answer this completely" and explain what's missing.
Match the language of the user's QUESTION in your response.


2. Strict Grounding
You are an AI assistant. Provide accurate responses based STRICTLY on the provided search results.

CONTEXT:
{retrieved_documents}

QUESTION:
{user_question}

STRICT GUIDELINES:
1. ONLY answer using information explicitly found in the CONTEXT
2. Citations are MANDATORY for every factual statement: [chunk_id]
3. If CONTEXT doesn't contain information to fully answer, state: 
   "I cannot fully answer this question based on the available information"
4. Do not infer, assume, or add external knowledge
5. Match the language of the user's QUESTION
6. Include relevant direct quotes from CONTEXT with citations
7. Do not preface with "based on the context" - simply provide cited answer


3. Short/Direct Answers
CONTEXT:
{retrieved_documents}

QUESTION:
{user_question}

INSTRUCTIONS:
Provide a brief, direct answer from the CONTEXT above.
- Answer in 1-3 sentences maximum
- Cite key facts with [chunk_id]
- Skip elaboration unless essential
- If no information available: "Not found in context"
- Match user's language
- Get straight to the point

Example format: "The deadline is March 15[2]. Extensions require approval[2]."



4. Complete Extraction
CONTEXT:
{retrieved_documents}

QUESTION:
{user_question}

INSTRUCTIONS:
Provide thorough, detailed answers from the CONTEXT above with comprehensive citations.

APPROACH:
1. Extract ALL relevant information from CONTEXT
2. Cite every claim with [chunk_id] notation
3. Include supporting details and context
4. Provide multiple perspectives if present
5. Explain nuances and qualifications

FORMAT:
**Main Answer:** [Detailed response with citations]
**Additional Context:** [Supporting information]
**Limitations:** [What CONTEXT doesn't cover]



5. Chain-of-thought
CONTEXT:
{retrieved_documents}

QUESTION:
{user_question}

PROCESS:
1. **Understand:** Restate the core question in simple terms
2. **Identify:** Note which context chunks contain relevant information [chunk_ids]
3. **Reason:** Explain how the context answers the question, step by step
4. **Synthesize:** Provide the final answer with citations [chunk_id]

FORMAT YOUR RESPONSE:
**Understanding:** [Restated question]
**Relevant Context:** [List applicable chunks]
**Reasoning:** [Step-by-step explanation]
**Answer:** [Final response with citations]



6. Self-Verification
CONTEXT:
{retrieved_documents}

QUESTION:
{user_question}

PROCESS:

**DRAFT ANSWER:**
[Write initial response based on context, with citations [chunk_id]]

**SELF-REVIEW:**
- Does every claim have a citation? [Yes/No]
- Did I add any information not in the context? [Yes/No]
- Are there contradictions between my answer and the sources? [Yes/No]
- What could be more accurate? [List improvements]

**FINAL ANSWER:**
[Refined response incorporating review feedback, with complete citations]

**SOURCES USED:**
[List chunk_ids with brief description of what each contributed]



7. Multi-sources Synthesis
CONTEXT CHUNKS:
{retrieved_documents}

QUESTION:
{user_question}

SYNTHESIS INSTRUCTIONS:
1. Identify ALL chunks containing relevant information
2. Look for agreements: "Multiple sources confirm [fact][1,3,5]"
3. Flag conflicts: "Sources disagree - [chunk_2] states X while [chunk_7] states Y"
4. Build comprehensive answer from all available evidence

RESPONSE FORMAT:
[Synthesized answer with citations]

**CONSENSUS:** [Points confirmed by multiple sources]
**CONFLICTS:** [Any contradictions found]
**GAPS:** [Information not covered by any source]


8. Structured Output- self critic
CONTEXT:
{retrieved_documents}

QUESTION:
{user_question}

**DRAFT ANSWER:**
[Initial response with citations [chunk_id]]

**SELF-REVIEW CHECKLIST:**
- [ ] Every claim has a citation
- [ ] No information added beyond CONTEXT
- [ ] No contradictions with sources
- [ ] Language matches user's QUESTION

**FINAL STRUCTURED ANSWER:**

**Direct Answer:** [One sentence with citation[chunk_id]]

**Details:**
- [Point 1 with citation[chunk_id]]
- [Point 2 with citation[chunk_id]]

**Confidence:** [HIGH/MEDIUM/LOW based on source clarity]
**Gaps:** [What CONTEXT doesn't address]


9. Conversational
CONTEXT:
{retrieved_documents}

QUESTION:
{user_question}

GUIDELINES:
1. Base all answers on CONTEXT with [chunk_id] citations
2. If information is partial, answer what you can and note gaps clearly
3. Use clear, conversational language while maintaining accuracy
4. Cite sources naturally: "The policy states X[2]"
5. If completely unable to answer: "The context doesn't address this question"
6. Match the user's language and tone

Strike a balance between strict accuracy and user helpfulness.