"""
Evaluator test cases: scripted interviews of KNOWN quality.

Each answer is written exactly as it would be spoken. Inline markers:
  [p2.5]   a 2.5-second pause
Fillers ("um", "uh"), stammers ("I-I"), repetitions and false starts ("we were— I built") are written as said.

`expect` gives the range a strict human recruiter would put the overall score in, plus extra checks.
"""

ROLE = "AI Engineer"
JD = ("AI Engineer (Mid). Build and ship production LLM applications: RAG pipelines, agent workflows, evaluation and "
      "monitoring, latency/cost optimisation. Python, FastAPI, vector databases, Kubernetes. Own features end to end "
      "and work with product teams.")
RESUME = ("2.5 years at a logistics company. Built an internal LLM platform (agents, prompt registry, LLM gateway), "
          "RAG over shipment SOPs, evals with LLM-as-judge, Kubernetes deployment. Python, FastAPI, Postgres/pgvector.")
NOTES = "Series B startup building AI copilots for customer support teams. Round: technical screen with the hiring manager."

QUESTIONS = [
    {"section": "role_alignment", "text": "Walk me through your background and why you're applying for this AI Engineer role."},
    {"section": "technical_background", "text": "Tell me about an LLM system you built end to end. How does it work?"},
    {"section": "technical_background", "text": "How did you evaluate and monitor the quality of that system in production?"},
    {"section": "impact_ownership", "text": "What was your personal contribution there, and what measurable impact did it have?"},
    {"section": "impact_ownership", "text": "Tell me about a time something broke in production. What did you do?"},
    {"section": "role_alignment", "text": "Why this company, and what would you bring to this team in your first 90 days?"},
]

CASES = [
    {
        "id": "A_excellent",
        "label": "Excellent: specific, first-person, quantified, fluent",
        "expect": {"overall": (65, 100)},
        "answers": [
            "I'm an AI engineer with two and a half years at a logistics company, where I moved from prototypes to owning "
            "production LLM systems. I'm applying because your role is exactly what I've been doing: shipping RAG and agent "
            "workflows end to end, with evals and cost control, and I want to do that for customer support, where response "
            "quality is directly measurable.",
            "I built a RAG assistant over about four thousand shipment SOP documents. Documents are chunked by section, "
            "embedded and stored in pgvector. At query time I do hybrid search, BM25 plus vectors, rerank the top twenty with "
            "a cross encoder, and pass the top five to the model with citations. A FastAPI service wraps it, behind our LLM "
            "gateway for rate limits and per-team token budgets. I chose hybrid search because pure vector search missed exact "
            "shipment codes.",
            "I built an offline eval set of three hundred real questions with reference answers, and scored retrieval with "
            "recall at five and answers with an LLM judge that I calibrated against two hundred human labels, about eighty "
            "seven percent agreement. Every prompt or model change runs that suite in CI. In production I log traces to "
            "Langfuse and track groundedness, latency and thumbs-down rate, with an alert if groundedness drops below ninety percent.",
            "I personally designed the retrieval pipeline and the eval harness and wrote most of the gateway. The impact: "
            "ops agents' average handling time dropped from eleven minutes to four, about forty percent of tickets are now "
            "resolved without escalation, and moving long-context calls to a cheaper model with caching cut our monthly LLM "
            "spend by sixty percent, roughly eighteen thousand dollars a month.",
            "A vendor model update started returning malformed JSON for about fifteen percent of tool calls, which broke our "
            "agent workflows. I caught it from the error-rate alert within ten minutes, rolled the gateway route back to the "
            "pinned model version, and then added schema validation with one automatic retry and a contract test in CI. Since "
            "then we pin model versions and canary every upgrade to five percent of traffic.",
            "You're building copilots for support teams, and that's the same problem I solved for ops agents: grounded "
            "answers, measurable quality, controlled cost. In my first ninety days I'd learn your ticket data and current "
            "metrics, ship an eval suite for your copilot if one doesn't exist, and take ownership of one retrieval or latency "
            "improvement with a clear target, like cutting p95 latency or raising deflection rate.",
        ],
    },
    {
        "id": "B_good_content_disfluent",
        "label": "Strong content, heavy fillers / stammers / pauses",
        "expect": {"overall": (45, 85), "communication_below": "A_excellent"},
        "answers": [
            "um so I'm I'm an AI engineer, uh, two and a half years at a logistics company, and um [p2.0] I-I moved from like "
            "prototypes to, uh, owning production LLM systems. And I'm applying because, you know, the role is basically what "
            "I've been doing, um, RAG and agents end to end with, uh, evals and cost control.",
            "so um [p1.8] I built a RAG assistant over, uh, about four thousand SOP documents. We— I chunk by section, embed into "
            "pgvector, and then um hybrid search, like BM25 plus vectors, then a cross encoder reranks the top twenty and, uh, "
            "the top five go to the model with citations. It's a FastAPI service behind our, um, LLM gateway. I picked hybrid "
            "because, uh, vector search was missing exact shipment codes, you know.",
            "um [p3.2] so for evals I-I built a set of three hundred real questions with, uh, reference answers, recall at five "
            "for retrieval and an LLM judge that I calibrated against, um, two hundred human labels, about eighty seven percent "
            "agreement. It runs in CI. And in production we we log traces to Langfuse and, uh, track groundedness and latency "
            "with an alert, like, below ninety percent.",
            "uh so I personally designed the the retrieval pipeline and the eval harness and um most of the gateway. Impact was "
            "um [p2.2] handling time went from eleven minutes to four, uh, forty percent of tickets resolved without escalation, "
            "and um cost went down sixty percent, around eighteen thousand dollars a month, basically.",
            "um yeah so a vendor model update started, uh, returning broken JSON for like fifteen percent of tool calls. I caught "
            "it from the alert in about ten minutes, rolled back to the pinned version, and then, um, added schema validation "
            "with a retry and a contract test. Now we, uh, pin versions and canary upgrades at five percent.",
            "um [p2.4] so you're building support copilots and that's, uh, basically the same problem I solved, grounded answers, "
            "measurable quality, cost. In ninety days I'd, um, learn your data and metrics, ship an eval suite if, uh, there "
            "isn't one, and own one improvement like, you know, p95 latency or deflection rate.",
        ],
    },
    {
        "id": "C_average_generic",
        "label": "Average: relevant but generic, 'we', few specifics",
        "expect": {"overall": (20, 55)},
        "answers": [
            "I have a computer science degree and I've been working in AI for about two years. I like working with LLMs and I "
            "think this role is a good fit because it's about building AI products, which is what I want to do.",
            "We built a chatbot that answers questions from documents. We used a vector database and an LLM, and the documents "
            "get split into chunks and then retrieved when a user asks something. It worked pretty well for the team.",
            "We tested it with some questions and checked if the answers looked right. We also had some logging so we could see "
            "if there were errors, and users could give feedback.",
            "I worked on the backend part and helped with the integration. The team was happy with it and it saved people some "
            "time when they were searching for information.",
            "One time the API was down and the chatbot stopped working. We waited for it to come back and then we added some "
            "error messages so users knew what was happening.",
            "I think the company is doing interesting things with AI. I would bring my experience with LLMs and I'm a fast "
            "learner, so I would try to learn the codebase and help the team.",
        ],
    },
    {
        "id": "D_weak_vague",
        "label": "Weak: short, vague, no specifics",
        "expect": {"overall": (0, 30)},
        "answers": [
            "I studied computer science and I want to work in AI.",
            "I made a chatbot using an API. It answered questions.",
            "I checked the answers manually.",
            "I did the coding part.",
            "I don't remember anything breaking.",
            "It seems like a good company. I'm interested in AI.",
        ],
    },
    {
        "id": "E_non_answers",
        "label": "Non-answers / confused / unprofessional (like the real bad session)",
        "expect": {"overall": (0, 15)},
        "answers": [
            "um [p3.8] what are you already talking? [p1.8] um how am I engineering? AI engineer, don't you have anything to "
            "say? [p1.7] I completed my graduation and I'm looking forward to work as AI engineer.",
            "Systems [p7.8] uh [p2.6] system.",
            "um",
            "I tend to use AI for editing videos.",
            "no",
            "whatever, next question.",
        ],
    },
    {
        "id": "F_fluent_but_wrong",
        "label": "Fluent and confident, technically wrong",
        "expect": {"overall": (0, 40), "technical_below": 35},
        "answers": [
            "I'm an AI engineer with two years of experience and I'm applying because I'm an expert in large language models "
            "and I know exactly how to make them reliable, which is what this role needs.",
            "I built a RAG system. RAG means we retrain the model every night on all our documents so it memorises them, which "
            "is why it never needs a database. Embeddings are just the model's temperature setting, and we set it to zero so "
            "the model has perfect memory of everything.",
            "Evaluation is easy: if temperature is zero the output is deterministic, and deterministic outputs are always "
            "correct, so we didn't need any evals or monitoring. Hallucinations only happen at high temperature.",
            "I owned the whole thing and the impact was huge, the model became one hundred percent accurate after we set "
            "temperature to zero, and accuracy is the only metric that matters.",
            "Nothing ever broke because LLMs are deterministic at temperature zero. If something did break, I'd just retrain "
            "the model overnight and it would fix itself.",
            "I'd bring my expertise in making models perfectly accurate, and in ninety days I'd retrain your model on all your "
            "support tickets so it never makes mistakes.",
        ],
    },
    {
        "id": "G_long_offtopic",
        "label": "Long rambling, off-topic (tries to game the length gates)",
        "expect": {"overall": (0, 25)},
        "answers": [
            "So my background, well I grew up in a small town and I always loved football, I used to play every weekend with "
            "my friends and we had this local team that was really good, and my favourite player was Messi because of how he "
            "dribbles, and I think football teaches you a lot about life in general and teamwork and things like that.",
            "An LLM system, yes, so I really enjoy using ChatGPT for planning my trips, last summer I went to the mountains and "
            "it helped me find hotels and restaurants, and the food there was amazing, especially the local cheese, and we "
            "also went hiking and the weather was perfect most of the days except one when it rained a lot.",
            "Monitoring quality, I think quality is very important in everything, like when I buy shoes I always check the "
            "quality of the stitching and the material, because cheap shoes break quickly, and my brother once bought shoes "
            "that broke after a week, which was really funny actually, we still joke about it.",
            "My contribution, I contribute a lot to my friend group, I am usually the one who organises the weekend plans and "
            "books the football pitch and collects money from everyone, which is a lot of work but somebody has to do it.",
            "Something broke, yes, my laptop broke last year, the screen cracked when I dropped it at the airport, and I had to "
            "get it repaired which took two weeks and cost a lot of money, so now I always use a protective case.",
            "Why this company, I think it is a nice company and the office is probably nice, and I would bring a positive "
            "attitude and I would organise team football matches on Fridays which would be great for team bonding.",
        ],
    },
]

# Re-run these to measure how stable the scores are between runs.
REPEATS = ["A_excellent", "D_weak_vague"]
