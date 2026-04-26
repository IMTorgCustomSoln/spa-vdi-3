

export class RecursiveCharacterTextSplitter {
    /*

    Key Logic Steps:
    * Hierarchy: It prioritizes \n\n to keep paragraphs together, then \n for sentences, and finally " " for words.
    * Recursion: If a string segment is still longer than chunkSize after splitting, it calls itself using the next available separator in the list.
    * Merging: It combines smaller splits back together until they just barely fit under the chunkSize limit.
    * Overlap: When moving to a new chunk, it "rewinds" and includes the last few segments from the previous chunk to maintain context.

    Usage:
        const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 50, chunkOverlap: 10 });
        const chunks = splitter.splitText("Your very long string...");

    */
  constructor({ chunkSize = 1000, chunkOverlap = 200, separators = ["\n\n", "\n", " ", ""] } = {}) {
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
    this.separators = separators;
  }

  splitText(text) {
    return this._recursiveSplit(text, this.separators);
  }

  _recursiveSplit(text, separators) {
    const finalChunks = [];
    let separator = separators[separators.length - 1];
    let nextSeparators = [];

    // Find the best separator that exists in the text
    for (let i = 0; i < separators.length; i++) {
      const s = separators[i];
      if (s === "") {
        separator = s;
        break;
      }
      if (text.includes(s)) {
        separator = s;
        nextSeparators = separators.slice(i + 1);
        break;
      }
    }

    const splits = separator !== "" ? text.split(separator) : text.split("");
    let goodSplits = [];

    for (const s of splits) {
      if (s.length < this.chunkSize) {
        goodSplits.push(s);
      } else {
        // If we have accumulated good splits, merge them before tackling the big one
        if (goodSplits.length > 0) {
          finalChunks.push(...this._mergeSplits(goodSplits, separator));
          goodSplits = [];
        }
        // Recursively split the part that is still too large
        finalChunks.push(...this._recursiveSplit(s, nextSeparators));
      }
    }

    if (goodSplits.length > 0) {
      finalChunks.push(...this._mergeSplits(goodSplits, separator));
    }

    return finalChunks;
  }

  _mergeSplits(splits, separator) {
    const docs = [];
    let currentDoc = [];
    let total = 0;

    for (const s of splits) {
      const len = s.length;
      const sepLen = currentDoc.length > 0 ? separator.length : 0;

      if (total + len + sepLen <= this.chunkSize) {
        currentDoc.push(s);
        total += len + sepLen;
      } else {
        if (currentDoc.length > 0) {
          docs.push(currentDoc.join(separator));
          
          // Rewind for overlap
          while (total > this.chunkOverlap || (total + len + sepLen > this.chunkSize && total > 0)) {
            const popped = currentDoc.shift();
            total -= popped.length + (currentDoc.length > 0 ? separator.length : 0);
          }
        }
        currentDoc.push(s);
        total += len + (currentDoc.length > 1 ? separator.length : 0);
      }
    }

    if (currentDoc.length > 0) {
      docs.push(currentDoc.join(separator));
    }
    return docs;
  }
}