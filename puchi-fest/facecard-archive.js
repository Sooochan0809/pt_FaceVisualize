(() => {
    "use strict";

    const ZIP_MIME_TYPE = "application/zip";
    const ZIP_UTF8_FLAG = 0x0800;
    const ZIP_STORE_METHOD = 0;
    const ZIP_VERSION = 20;
    const ZIP_LOCAL_HEADER_SIGNATURE = 0x04034b50;
    const ZIP_CENTRAL_HEADER_SIGNATURE = 0x02014b50;
    const ZIP_END_SIGNATURE = 0x06054b50;
    const ZIP_LOCAL_HEADER_SIZE = 30;
    const ZIP_CENTRAL_HEADER_SIZE = 46;
    const ZIP_END_SIZE = 22;
    const ZIP_MAX_COMMENT_SIZE = 0xffff;
    const ARCHIVE_SCHEMA_VERSION = 1;
    const ARCHIVE_FILE_SUFFIX = ".facecard.zip";
    const textEncoder = new TextEncoder();
    const textDecoder = new TextDecoder();
    const crcTable = createCrcTable();

    function createCrcTable() {
        return Uint32Array.from({ length: 256 }, (_, value) => {
            let crc = value;
            for (let bit = 0; bit < 8; bit++) {
                crc = crc & 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
            }
            return crc >>> 0;
        });
    }

    function crc32(bytes) {
        let crc = 0xffffffff;
        for (const byte of bytes) {
            crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
        }
        return (crc ^ 0xffffffff) >>> 0;
    }

    function zipDateTime(date) {
        const year = Math.max(1980, date.getFullYear());
        return {
            date:
                ((year - 1980) << 9)
                | ((date.getMonth() + 1) << 5)
                | date.getDate(),
            time:
                (date.getHours() << 11)
                | (date.getMinutes() << 5)
                | Math.floor(date.getSeconds() / 2)
        };
    }

    function createHeader(size, write) {
        const bytes = new Uint8Array(size);
        write(new DataView(bytes.buffer));
        return bytes;
    }

    function createZip(entries, createdAt) {
        const timestamp = zipDateTime(createdAt);
        const prepared = entries.map(entry => {
            const nameBytes = textEncoder.encode(entry.name);
            return {
                ...entry,
                nameBytes,
                crc: crc32(entry.bytes),
                localOffset: 0
            };
        });
        const localParts = [];
        let localSize = 0;

        for (const entry of prepared) {
            entry.localOffset = localSize;
            const header = createHeader(
                ZIP_LOCAL_HEADER_SIZE + entry.nameBytes.length,
                view => {
                    view.setUint32(0, ZIP_LOCAL_HEADER_SIGNATURE, true);
                    view.setUint16(4, ZIP_VERSION, true);
                    view.setUint16(6, ZIP_UTF8_FLAG, true);
                    view.setUint16(8, ZIP_STORE_METHOD, true);
                    view.setUint16(10, timestamp.time, true);
                    view.setUint16(12, timestamp.date, true);
                    view.setUint32(14, entry.crc, true);
                    view.setUint32(18, entry.bytes.length, true);
                    view.setUint32(22, entry.bytes.length, true);
                    view.setUint16(26, entry.nameBytes.length, true);
                    view.setUint16(28, 0, true);
                    new Uint8Array(view.buffer, ZIP_LOCAL_HEADER_SIZE)
                        .set(entry.nameBytes);
                }
            );
            localParts.push(header, entry.bytes);
            localSize += header.length + entry.bytes.length;
        }

        const centralParts = [];
        let centralSize = 0;
        for (const entry of prepared) {
            const header = createHeader(
                ZIP_CENTRAL_HEADER_SIZE + entry.nameBytes.length,
                view => {
                    view.setUint32(0, ZIP_CENTRAL_HEADER_SIGNATURE, true);
                    view.setUint16(4, ZIP_VERSION, true);
                    view.setUint16(6, ZIP_VERSION, true);
                    view.setUint16(8, ZIP_UTF8_FLAG, true);
                    view.setUint16(10, ZIP_STORE_METHOD, true);
                    view.setUint16(12, timestamp.time, true);
                    view.setUint16(14, timestamp.date, true);
                    view.setUint32(16, entry.crc, true);
                    view.setUint32(20, entry.bytes.length, true);
                    view.setUint32(24, entry.bytes.length, true);
                    view.setUint16(28, entry.nameBytes.length, true);
                    view.setUint16(30, 0, true);
                    view.setUint16(32, 0, true);
                    view.setUint16(34, 0, true);
                    view.setUint16(36, 0, true);
                    view.setUint32(38, 0, true);
                    view.setUint32(42, entry.localOffset, true);
                    new Uint8Array(view.buffer, ZIP_CENTRAL_HEADER_SIZE)
                        .set(entry.nameBytes);
                }
            );
            centralParts.push(header);
            centralSize += header.length;
        }

        const end = createHeader(ZIP_END_SIZE, view => {
            view.setUint32(0, ZIP_END_SIGNATURE, true);
            view.setUint16(4, 0, true);
            view.setUint16(6, 0, true);
            view.setUint16(8, prepared.length, true);
            view.setUint16(10, prepared.length, true);
            view.setUint32(12, centralSize, true);
            view.setUint32(16, localSize, true);
            view.setUint16(20, 0, true);
        });
        return new Blob([...localParts, ...centralParts, end], {
            type: ZIP_MIME_TYPE
        });
    }

    async function urlEntry(name, url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`${name}を読み込めませんでした。`);
        return blobEntry(name, await response.blob());
    }

    async function blobEntry(name, blob) {
        return {
            name,
            bytes: new Uint8Array(await blob.arrayBuffer()),
            type: blob.type || mimeTypeForName(name)
        };
    }

    function mimeTypeForName(name) {
        if (/\.png$/i.test(name)) return "image/png";
        if (/\.webp$/i.test(name)) return "image/webp";
        if (/\.jpe?g$/i.test(name)) return "image/jpeg";
        if (/\.webm$/i.test(name)) return "video/webm";
        if (/\.json$/i.test(name)) return "application/json";
        return "application/octet-stream";
    }

    function imageExtension(type) {
        if (type === "image/png") return "png";
        if (type === "image/jpeg") return "jpg";
        return "webp";
    }

    function safeArchiveId(archiveId) {
        const safe = String(archiveId).replace(/[^0-9A-Za-z_-]/g, "");
        if (!safe) throw new Error("アーカイブ番号が不正です。");
        return safe;
    }

    function archiveImagePath(folder, index, extension) {
        const fileNumber = String(index + 1).padStart(2, "0");
        return `${folder}/${fileNumber}.${extension}`;
    }

    async function createImageEntries(root, folder, images) {
        return Promise.all(images.map(async (item, index) => {
            const provisionalPath = archiveImagePath(folder, index, "webp");
            const source = await urlEntry(
                `${root}${provisionalPath}`,
                item.imageUrl
            );
            return {
                ...source,
                name: `${root}${archiveImagePath(
                    folder,
                    index,
                    imageExtension(source.type)
                )}`
            };
        }));
    }

    function createImageManifest(root, images, entries, selected) {
        return images.map((item, index) => ({
            file: entries[index].name.slice(root.length),
            selected,
            selectionOrder: selected ? index + 1 : null,
            capturedAtSeconds: Number(item.time) || 0,
            emotion: item.emotion || "",
            emotionScore: Number(item.emotionScore) || 0,
            frontalScore: Number(item.frontalScore) || 0
        }));
    }

    async function create({
        archiveId,
        createdAt = new Date(),
        lenticular,
        selectedImages,
        unselectedImages = [],
        experienceVideo = null,
        cardArtUrl
    }) {
        const safeId = safeArchiveId(archiveId);
        if (!lenticular?.imageUrl) throw new Error("レンチキュラー画像がありません。");
        if (!Array.isArray(selectedImages) || !selectedImages.length) {
            throw new Error("選択画像がありません。");
        }
        if (!cardArtUrl) throw new Error("カード台紙がありません。");

        const root = `${safeId}/`;
        const lenticularEntry = await urlEntry(
            `${root}lenticular.png`,
            lenticular.imageUrl
        );
        const cardArtEntry = await urlEntry(`${root}card-art.png`, cardArtUrl);
        const [selectedEntries, unselectedEntries] = await Promise.all([
            createImageEntries(root, "highlights/selected", selectedImages),
            createImageEntries(root, "highlights/unselected", unselectedImages)
        ]);
        const videoEntry = experienceVideo?.blob instanceof Blob
            ? await blobEntry(`${root}experience.webm`, experienceVideo.blob)
            : null;
        const manifest = {
            schemaVersion: ARCHIVE_SCHEMA_VERSION,
            archiveId: safeId,
            createdAt: createdAt.toISOString(),
            cardTemplate: "suitopia-card-v1",
            cardArt: "card-art.png",
            lenticular: {
                file: "lenticular.png",
                width: lenticular.width,
                height: lenticular.height,
                settings: lenticular.settings
            },
            experienceVideo: videoEntry ? {
                file: "experience.webm",
                mimeType: experienceVideo.mimeType || videoEntry.type,
                durationSeconds: Math.max(0, Number(experienceVideo.durationMs) || 0) / 1000,
                width: Number(experienceVideo.width) || 0,
                height: Number(experienceVideo.height) || 0,
                audio: false
            } : null,
            selectedImages: createImageManifest(
                root,
                selectedImages,
                selectedEntries,
                true
            ),
            unselectedImages: createImageManifest(
                root,
                unselectedImages,
                unselectedEntries,
                false
            )
        };
        const manifestEntry = {
            name: `${root}manifest.json`,
            bytes: textEncoder.encode(JSON.stringify(manifest, null, 2))
        };
        return {
            blob: createZip(
                [
                    manifestEntry,
                    lenticularEntry,
                    cardArtEntry,
                    ...(videoEntry ? [videoEntry] : []),
                    ...selectedEntries,
                    ...unselectedEntries
                ],
                createdAt
            ),
            fileName: `${safeId}${ARCHIVE_FILE_SUFFIX}`,
            manifest
        };
    }

    function download(blob, fileName) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = fileName;
        document.body.append(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function findEndRecord(view) {
        const minimumOffset = Math.max(
            0,
            view.byteLength - ZIP_END_SIZE - ZIP_MAX_COMMENT_SIZE
        );
        for (let offset = view.byteLength - ZIP_END_SIZE; offset >= minimumOffset; offset--) {
            if (view.getUint32(offset, true) === ZIP_END_SIGNATURE) return offset;
        }
        throw new Error("ZIPの終端情報が見つかりません。");
    }

    function readEntries(arrayBuffer) {
        const view = new DataView(arrayBuffer);
        const endOffset = findEndRecord(view);
        const entryCount = view.getUint16(endOffset + 10, true);
        let centralOffset = view.getUint32(endOffset + 16, true);
        const entries = new Map();

        for (let index = 0; index < entryCount; index++) {
            if (view.getUint32(centralOffset, true) !== ZIP_CENTRAL_HEADER_SIGNATURE) {
                throw new Error("ZIPのファイル一覧が壊れています。");
            }
            const method = view.getUint16(centralOffset + 10, true);
            if (method !== ZIP_STORE_METHOD) {
                throw new Error("このZIPの圧縮方式には対応していません。");
            }
            const expectedCrc = view.getUint32(centralOffset + 16, true);
            const compressedSize = view.getUint32(centralOffset + 20, true);
            const fileNameLength = view.getUint16(centralOffset + 28, true);
            const extraLength = view.getUint16(centralOffset + 30, true);
            const commentLength = view.getUint16(centralOffset + 32, true);
            const localOffset = view.getUint32(centralOffset + 42, true);
            const fileName = textDecoder.decode(
                new Uint8Array(
                    arrayBuffer,
                    centralOffset + ZIP_CENTRAL_HEADER_SIZE,
                    fileNameLength
                )
            );
            if (view.getUint32(localOffset, true) !== ZIP_LOCAL_HEADER_SIGNATURE) {
                throw new Error(`${fileName}の情報が壊れています。`);
            }
            const localNameLength = view.getUint16(localOffset + 26, true);
            const localExtraLength = view.getUint16(localOffset + 28, true);
            const dataOffset =
                localOffset
                + ZIP_LOCAL_HEADER_SIZE
                + localNameLength
                + localExtraLength;
            const bytes = new Uint8Array(
                arrayBuffer.slice(dataOffset, dataOffset + compressedSize)
            );
            if (crc32(bytes) !== expectedCrc) {
                throw new Error(`${fileName}の検証に失敗しました。`);
            }
            entries.set(fileName, bytes);
            centralOffset +=
                ZIP_CENTRAL_HEADER_SIZE
                + fileNameLength
                + extraLength
                + commentLength;
        }
        return entries;
    }

    function findArchiveEntry(entries, root, relativePath) {
        const entry = entries.get(`${root}${relativePath}`);
        if (!entry) throw new Error(`${relativePath}がアーカイブにありません。`);
        return entry;
    }

    function createEntryUrl(entries, root, relativePath) {
        const bytes = findArchiveEntry(entries, root, relativePath);
        return URL.createObjectURL(new Blob([bytes], {
            type: mimeTypeForName(relativePath)
        }));
    }

    async function read(file) {
        const entries = readEntries(await file.arrayBuffer());
        const manifestName = [...entries.keys()].find(name =>
            name.endsWith("/manifest.json") || name === "manifest.json"
        );
        if (!manifestName) throw new Error("manifest.jsonがありません。");
        const root = manifestName.slice(0, -"manifest.json".length);
        const manifest = JSON.parse(textDecoder.decode(entries.get(manifestName)));
        if (manifest.schemaVersion !== ARCHIVE_SCHEMA_VERSION) {
            throw new Error("対応していないアーカイブ形式です。");
        }

        const urls = [];
        const entryUrl = relativePath => {
            const url = createEntryUrl(entries, root, relativePath);
            urls.push(url);
            return url;
        };
        const lenticularImageUrl = entryUrl(manifest.lenticular.file);
        const cardArtUrl = entryUrl(manifest.cardArt);
        const restoreImages = items => items.map(item => ({
            ...item,
            imageUrl: entryUrl(item.file)
        }));
        const selectedImages = restoreImages(manifest.selectedImages);
        const unselectedImages = restoreImages(manifest.unselectedImages ?? []);
        const experienceVideoUrl = manifest.experienceVideo
            ? entryUrl(manifest.experienceVideo.file)
            : "";

        return {
            manifest,
            lenticularImageUrl,
            cardArtUrl,
            selectedImages,
            unselectedImages,
            experienceVideoUrl,
            revoke() {
                urls.forEach(url => URL.revokeObjectURL(url));
            }
        };
    }

    window.MotionFacecardArchive = Object.freeze({
        create,
        download,
        read
    });
})();
