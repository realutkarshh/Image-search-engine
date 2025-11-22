package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/joho/godotenv"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

/*
	==============================
	  IMAGE CRAWLER CONFIG
	==============================
*/

const (
	MaxImagePages     = 400
	ImageTimeout      = 9 * time.Second
	ImageDelay        = 350 * time.Millisecond
	MaxImageBodySize  = 3 * 1024 * 1024
	MaxImageDepth     = 4
)

/*
	==============================
	  MONGODB STRUCT
	==============================
*/

type ImageRecord struct {
	FileURL     string    `bson:"file_url"`
	AltText     string    `bson:"alt_text"`
	CaptionText string    `bson:"caption_text"`
	PageURL     string    `bson:"page_url"`
	DomainName  string    `bson:"domain_name"`
	Format      string    `bson:"format"`
	Width       string    `bson:"width"`
	Height      string    `bson:"height"`
	TimeFetched time.Time `bson:"time_fetched"`
}

/*
	==============================
	   ENV HELPERS
	==============================
*/

func readEnv(key, fallback string) string {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	return v
}

/*
	==============================
	   DOMAIN & URL HELPERS
	==============================
*/

func domainAllowed(parsed *url.URL, allowed []string) bool {
	host := parsed.Hostname()
	for _, d := range allowed {
		if strings.HasSuffix(host, d) {
			return true
		}
	}
	return false
}

func resolveURL(base *url.URL, href string) (*url.URL, error) {
	if strings.TrimSpace(href) == "" {
		return nil, fmt.Errorf("empty link")
	}
	ref, err := url.Parse(href)
	if err != nil {
		return nil, err
	}
	if !ref.IsAbs() {
		ref = base.ResolveReference(ref)
	}
	if ref.Scheme != "http" && ref.Scheme != "https" {
		return nil, fmt.Errorf("bad scheme")
	}
	ref.Fragment = ""
	return ref, nil
}

/*
	==============================
	   MONGO CONNECTION
	==============================
*/

func initImageDB(ctx context.Context) (*mongo.Client, *mongo.Collection, error) {
	uri := readEnv("IMG_DB_URI", "")
	db := readEnv("IMG_DB_NAME", "image_indexer_db")

	if uri == "" {
		return nil, nil, fmt.Errorf("IMG_DB_URI not provided")
	}

	clientOpts := options.Client().ApplyURI(uri)
	client, err := mongo.Connect(ctx, clientOpts)
	if err != nil {
		return nil, nil, err
	}
	if err := client.Ping(ctx, nil); err != nil {
		return nil, nil, err
	}

	collection := client.Database(db).Collection("image_files")
	return client, collection, nil
}

func saveImage(ctx context.Context, col *mongo.Collection, img ImageRecord) error {
	filter := bson.M{"file_url": img.FileURL}
	update := bson.M{"$set": img}
	opts := options.Update().SetUpsert(true)

	_, err := col.UpdateOne(ctx, filter, update, opts)
	return err
}

/*
	==============================
	   FETCH HTML PAGE
	==============================
*/

func downloadHTML(link string) (*goquery.Document, error) {
	client := &http.Client{Timeout: ImageTimeout}

	resp, err := client.Get(link)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if !strings.Contains(resp.Header.Get("Content-Type"), "text/html") {
		return nil, fmt.Errorf("not html content")
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, MaxImageBodySize))
	if err != nil {
		return nil, err
	}

	return goquery.NewDocumentFromReader(bytes.NewReader(body))
}

/*
	==============================
	  IMAGE FORMAT FILTER
	==============================
*/

func isAllowedImageFormat(src string) bool {
	src = strings.ToLower(src)

	if strings.HasPrefix(src, "data:") {
		return false
	}

	allowed := []string{".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp"}

	for _, ext := range allowed {
		if strings.HasSuffix(src, ext) {
			return true
		}
	}

	return false
}

/*
	==============================
	  IMAGE EXTRACTION LOGIC
	==============================
*/

func parseImages(page string, doc *goquery.Document) []ImageRecord {
	base, _ := url.Parse(page)
	domain := base.Hostname()

	var out []ImageRecord

	doc.Find("img").Each(func(i int, tag *goquery.Selection) {

		// Check all possible lazy-load attributes
		candidates := []string{"src", "data-src", "data-lazy-src", "data-original", "data-img", "data-image"}

		var rawSrc string
		for _, a := range candidates {
			if v, ok := tag.Attr(a); ok && v != "" {
				rawSrc = v
				break
			}
		}

		if rawSrc == "" {
			return
		}

		imgURL, err := url.Parse(rawSrc)
		if err != nil {
			return
		}

		if !imgURL.IsAbs() {
			imgURL = base.ResolveReference(imgURL)
		}

		finalURL := imgURL.String()

		// EXTENSION FILTER
		if !isAllowedImageFormat(finalURL) {
			return
		}

		alt, _ := tag.Attr("alt")
		w, _ := tag.Attr("width")
		h, _ := tag.Attr("height")

		// detect file extension
		ext := ""
		lower := strings.ToLower(finalURL)
		switch {
		case strings.Contains(lower, ".jpg"), strings.Contains(lower, ".jpeg"):
			ext = "jpg"
		case strings.Contains(lower, ".png"):
			ext = "png"
		case strings.Contains(lower, ".webp"):
			ext = "webp"
		case strings.Contains(lower, ".gif"):
			ext = "gif"
		case strings.Contains(lower, ".avif"):
			ext = "avif"
		case strings.Contains(lower, ".bmp"):
			ext = "bmp"
		}

		// capture figcaption
		caption := ""
		if parentFig := tag.ParentsFiltered("figure"); parentFig.Length() > 0 {
			caption = strings.TrimSpace(parentFig.Find("figcaption").Text())
		}

		out = append(out, ImageRecord{
			FileURL:     finalURL,
			AltText:     alt,
			CaptionText: caption,
			PageURL:     page,
			DomainName:  domain,
			Format:      ext,
			Width:       w,
			Height:      h,
			TimeFetched: time.Now().UTC(),
		})
	})

	return out
}

/*
	==============================
	   IMAGE CRAWLING ENGINE
	==============================
*/

func runImageCrawler(ctx context.Context, col *mongo.Collection) error {
	seedEnv := readEnv("IMG_SEED_LINKS", "")
	if seedEnv == "" {
		return fmt.Errorf("IMG_SEED_LINKS is empty")
	}

	seeds := strings.Split(seedEnv, ",")

	domainEnv := readEnv("IMG_ALLOWED_SITES", "")
	allowed := []string{}
	if domainEnv != "" {
		for _, d := range strings.Split(domainEnv, ",") {
			allowed = append(allowed, strings.TrimSpace(d))
		}
	}

	type Task struct {
		Link  string
		Level int
	}

	queue := []Task{}
	seen := map[string]bool{}

	for _, s := range seeds {
		if strings.TrimSpace(s) != "" {
			queue = append(queue, Task{Link: s, Level: 0})
		}
	}

	processed := 0

	for len(queue) > 0 && processed < MaxImagePages {
		t := queue[0]
		queue = queue[1:]

		if seen[t.Link] {
			continue
		}
		seen[t.Link] = true

		parsed, err := url.Parse(t.Link)
		if err != nil {
			continue
		}
		if !domainAllowed(parsed, allowed) {
			continue
		}

		log.Println("Fetching:", t.Link)
		doc, err := downloadHTML(t.Link)
		if err != nil {
			log.Println("ERROR:", err)
			continue
		}

		// extract filtered images
		found := parseImages(t.Link, doc)
		log.Printf("Found %d valid images on %s", len(found), t.Link)

		for _, img := range found {
			saveImage(ctx, col, img)
		}

		processed++
		log.Printf("Processed %d pages", processed)

		// follow links
		if t.Level < MaxImageDepth {
			doc.Find("a[href]").Each(func(i int, a *goquery.Selection) {
				raw, _ := a.Attr("href")
				resolved, err := resolveURL(parsed, raw)
				if err == nil && !seen[resolved.String()] {
					queue = append(queue, Task{
						Link:  resolved.String(),
						Level: t.Level + 1,
					})
				}
			})
		}

		time.Sleep(ImageDelay)
	}

	return nil
}

/*
	==============================
	      MAIN
	==============================
*/

func main() {
	godotenv.Load()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	client, col, err := initImageDB(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer client.Disconnect(ctx)

	if err := runImageCrawler(ctx, col); err != nil {
		log.Fatal(err)
	}
}
