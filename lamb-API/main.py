from pathlib import Path
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from pydantic import BaseModel

load_dotenv()
client = OpenAI()
app = FastAPI()

origins = [
  "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReferencesRequest(BaseModel):
  transcription: str

class SummaryRequest(BaseModel):
  text: str
  system_prompt: str

# bible > Book > chapter > verse
class BibleReference(BaseModel):
  translation: str | None = None
  book: str
  chapter: int
  verse_start: int
  verse_end: int | None = None  # None = single verse

class ReferencesResponse(BaseModel):
  references: list[BibleReference]

REF_PROMPT = """
Given the following text, extract the Bible references mentioned.

Rules:
- Use canonical book names (e.g. Mark, not Mk).
- For a single verse, set verse_start only and leave verse_end null.
- For a range (e.g. Mark 6:35–44), set verse_start and verse_end.
- Only include references explicitly cited in the text.
- Luke 11:2–4 is included: "Luke 11:2–4, Jesus again teaches His disciples to pray to the Father, asking for daily bread, forgiveness, and protection from temptation." but the reference was not explicitly mentioned in the text so detecting those references are expected as well as important.
- set translation based on the translation you used as context or are referencing from, unless mentioned in the text
- A bible verse might be mentioned in the text without being explicitly cited, so detecting those references are expected as well as important.

## Example

Text: "“Good morning, church!” Pastor James said, smiling from the pulpit. “It is a blessing to see all of you today. Some of you look happy to be here, and some of you look like your alarm clock personally offended you this morning.”

The congregation laughed.

“Well, anyway, let’s get into the Word.”

Pastor James opened his Bible. “John 1:1 tells us, ‘In the beginning was the Word, and the Word was with God, and the Word was God.’ Before anything else existed, God was already there.”

He paused. “Now, speaking of beginnings, I tried making breakfast this morning. I burned the toast so badly the smoke alarm started praising the Lord before I did.”

Everyone laughed again.

“Now where was I? Oh yes—the Bible.”

He turned to Matthew 14:15, where the disciples noticed it was getting late and told Jesus to send the crowds away so they could find food.

“This same miracle is described in Mark 6:35–44. Thousands of people were hungry, but Jesus took five loaves and two fish, blessed them, and fed everyone. There were even twelve baskets left over.”

Pastor James looked around the room. “Sometimes we look at what we have and say, ‘Lord, this isn’t enough.’ But God can do a lot with a little.”

He stopped for a second. “That reminds me—I went fishing with my uncle once. We spent six hours on the water and caught absolutely nothing. He still came home telling everybody, ‘You should’ve seen the one that got away.’ Apparently, that fish was about the size of a school bus.”

The church laughed.

“Anyway, that has nothing to do with my sermon.”

Pastor James continued. “Jesus also taught us how to pray. In Matthew 6:9–13, He gives us the Lord’s Prayer: ‘Our Father which art in heaven, Hallowed be thy name… Give us this day our daily bread…’”

“And, Jesus again teaches His disciples to pray to the Father, asking for daily bread, forgiveness, and protection from temptation.”

He closed his Bible.

“So church, remember this: Jesus is the Word, Jesus provides what we need, and Jesus teaches us to pray.”

Pastor James smiled. “And if you forget everything else I said today, at least remember this: never let me cook breakfast for the church.”

The congregation laughed.

“Amen?”
“Amen!”"

Output:
{
  "references": [
    {
      "translation": null,
      "book": "John",
      "chapter": 1,
      "verse_start": 1,
      "verse_end": null
    },
    {
      "translation": null,
      "book": "Matthew",
      "chapter": 14,
      "verse_start": 15,
      "verse_end": null
    },
    {
      "translation": null,
      "book": "Mark",
      "chapter": 6,
      "verse_start": 35,
      "verse_end": 44
    },
    {
      "translation": null,
      "book": "Matthew",
      "chapter": 6,
      "verse_start": 9,
      "verse_end": 13
    },
    {
      "translation": null,
      "book": "Luke",
      "chapter": 11,
      "verse_start": 2,
      "verse_end": 4
    }
  ]
}
""".strip()

@app.get("/")
def root():
    return {"Bye": "Hello from Project Lamb API"}

@app.post("/references")
async def references(request: ReferencesRequest):

  try:
    response = client.responses.parse(
      model="gpt-4o-mini",
      input=[
          {
            "role": "system", 
            "content": REF_PROMPT
          },
          {
            "role": "user",
            "content": request.transcription
          }
      ],
      text_format=ReferencesResponse,
    )
  except Exception as e:
    return {"error": str(e)}

  return response.output_parsed


@app.post("/summarize")
async def summarize(request: SummaryRequest):
  return {"summary": "This is a summary of the text", "system_prompt": request.system_prompt}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

  results = {}
  model = "gpt-transcribe"

  suffix = Path(file.filename or "upload").suffix or ".bin"
  contents = await file.read()

  with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(contents)
    input_path = tmp.name

  try:
    audio = AudioSegment.from_file(input_path)
    ten_minutes = 10 * 60 * 1000
    first_10_minutes = audio[:ten_minutes]
    first_10_minutes.export("temp.wav", format="wav")

    with open("temp.wav", "rb") as audio_file:
      response = client.audio.transcriptions.create(
        model=model,
        file=audio_file,
        extra_body={
          "keywords": ["Bible", "King James Version", "Jesus", "Lamb", "Ark"],
          "languages": ["en"],
        },
      )
  finally:
    Path(input_path).unlink(missing_ok=True)

    results.update({
      "model": model,
      "results": response
    })

    return results