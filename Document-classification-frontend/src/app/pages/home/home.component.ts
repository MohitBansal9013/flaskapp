import { Component, Directive, EventEmitter, Input, OnInit, Output, QueryList, ViewChild, ViewChildren } from '@angular/core';

import { ModalDismissReasons, NgbModal, NgbModalRef } from '@ng-bootstrap/ng-bootstrap';
import { HomeService } from './home.service';
import { FlashMessagesService } from 'flash-messages-angular';

interface Cluster {
  id: number;
  tags: string;
  number_of_paragraphs: number;
  match_probability: number;
  manually_tagged: string;
  paragraphs: any
}

const CLUSTERS: Cluster[] = [
  {
    'id': 0,
    'tags': 'Loading...',
    'number_of_paragraphs': 0,
    'match_probability': 0,
    'manually_tagged': 'Loading...',
    'paragraphs': []
  }]
const error = { "message": '', "statuscode": 0 }
export type SortColumn = keyof Cluster | '';
export type SortDirection = 'asc' | 'desc' | '';
const rotate: { [key: string]: SortDirection } = { 'asc': 'desc', 'desc': '', '': 'asc' };
const compare = (v1: string | number, v2: string | number) => v1 < v2 ? -1 : v1 > v2 ? 1 : 0;
export interface SortEvent {
  column: SortColumn;
  direction: SortDirection;
}

@Directive({
  selector: 'th[sortable]',
  host: {
    '[class.asc]': 'direction === "asc"',
    '[class.desc]': 'direction === "desc"',
    '(click)': 'rotate()'
  }
})

export class NgbdSortableHeader {

  @Input() sortable: SortColumn = '';
  @Input() direction: SortDirection = '';
  @Output() sort = new EventEmitter<SortEvent>();

  rotate() {
    this.direction = rotate[this.direction];
    this.sort.emit({ column: this.sortable, direction: this.direction });
  }
}

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css'],
  providers: [HomeService]
})
export class HomeComponent implements OnInit {

  paragraphsIconSrc = 'assets/img/paragraphs-icon.png'
  paragraphsTitleTxt = 'Click See Paragraphs'
  wordCloudTxt = "Click See Word Cloud"
  closeModal: string = ''
  graphSrc: string = ""
  resetSrc = 'assets/img/reset-icon.png'
  wordCloudIconSrc = 'assets/img/word-cloud-icon.png'
  loaderSrc = 'assets/img/loader.gif'
  clusterWordCloud = ''
  page = 1
  pageSize = 4
  maxSize = 10
  tags: Array<string> = []
  clusters = CLUSTERS
  @ViewChildren(NgbdSortableHeader) headers!: QueryList<NgbdSortableHeader>;
  selectedCluster: any = ''
  HighlightRow: Number = 0;
  loading: boolean = false;
  loadingClusterWordCloud: boolean = false;
  loadingTaggingCluster: boolean = false;
  selectedTag: any = []
  customeTag: string = ''
  error = error
  addingTag = false
  isTagModalOpen = false
  tagModalRef!: NgbModalRef;
  
  constructor(private modalService: NgbModal,
    private homeService: HomeService,
    private flashMessagesService: FlashMessagesService) {

  }


  ngOnInit(): void {
    this.loading = true;
    this.graphSrc = ''
    this.clusters = CLUSTERS
    this.tags = ["Choose..."]
    this.error = error
    this.homeService.getTags().subscribe({
      next: (response) => {
        this.tags = [...this.tags, ...response.data]
      },
      error: (error) => {
        console.log('Request failed with error')
        console.log(error)
      },
      complete: () => {
        console.error('Request completed')
      }
    })

    this.homeService.getInitialClusterList().subscribe({
      next: (response) => {
        this.clusters = response.cluster_info
        console.log("response", response)
        this.clusters.forEach(cluster => {
          this.selectedTag[cluster.id] = this.tags[0]
        })
      },
      error: (error) => {
        console.log("error", error)
      },
      complete: () => {
        console.log("complete")
      }
    })

    this.clusterDetail(this.clusters[0])

  }

  onSort({ column, direction }: SortEvent) {

    // resetting other headers
    this.headers.forEach(header => {
      if (header.sortable !== column) {
        header.direction = '';
      }
    });

    // sorting countries
    if (direction === '' || column === '') {
      this.clusters = this.clusters;
    } else {
      this.clusters = [...this.clusters].sort((a, b) => {
        const res = compare(a[column], b[column]);
        return direction === 'asc' ? res : -res;
      });
    }
  }

  triggerModal(content: any) {
    this.modalService.open(content, { size: 'xl', ariaLabelledBy: 'modal-basic-title' }).result.then((res) => {
      this.closeModal = `Closed with: ${res}`;
    }, (res) => {
      this.closeModal = `Dismissed ${this.getDismissReason(res)}`;
    });
  }

  private getDismissReason(reason: any): string {
    if (reason === ModalDismissReasons.ESC) {
      return 'by pressing ESC';
    } else if (reason === ModalDismissReasons.BACKDROP_CLICK) {
      return 'by clicking on a backdrop';
    } else {
      return `with: ${reason}`;
    }
  }

  clusterDetail(cluster: Cluster) {
    this.loadingClusterWordCloud = true
    this.selectedCluster = cluster
    console.log(this.selectedCluster.id)
    this.HighlightRow = this.selectedCluster.id
    this.homeService.getWordCloud(this.selectedCluster.id).subscribe({
      next: (response) => {
        console.log(response)
        this.clusterWordCloud = response.data
      },
      error: (error) => {
        console.log(error)
      },
      complete: () => {
        console.log("done")
        this.loadingClusterWordCloud = false
      }
    })
  }

  tagCluster(clusterId: number) {
    let _selectedTag = this.selectedTag[clusterId]
    if (_selectedTag == this.tags[0]) {
      alert("Please select Tag")
      return false
    }
    this.loadingTaggingCluster = true
    this.homeService.tagCluster(clusterId, _selectedTag).subscribe({
      next: (response) => {
        this.clusters = response.cluster_info
        this.graphSrc = response.graph
      },
      error: (error) => {
        console.log("error", error)
      },
      complete: () => {
        this.loadingTaggingCluster = false
        console.log("complete")
      }
    })
    return false
  }

  reset() {
    this.ngOnInit()
  }

  openTagPopup(content: any) {
    this.isTagModalOpen = true
    this.error = error
    this.tagModalRef = this.modalService.open(content,{ ariaLabelledBy: 'modal-basic-title', windowClass: 'my-class' })
    this.tagModalRef.result.then((res) => {
      this.closeModal = `Closed with: ${res}`;
    }, (res) => {
      this.closeModal = `Dismissed ${this.getDismissReason(res)}`;
      this.isTagModalOpen = false
    });
  }

  addTag() {
    this.addingTag = true
    this.homeService.addTag(this.customeTag).subscribe({
      next: (response) => {
        if (response.statuscode == 400) {
          this.error = response
        } else {
          this.tags = ["Choose..."]
          this.tags = [...this.tags, ...response.data]
          this.tagModalRef.close()
          this.customeTag = ''
          this.flashMessagesService.show('Tag added successfully!!', { cssClass: 'alert-success', timeout: 2000 });
        }
      },
      error: (error) => {
        console.log("error", error)
      },
      complete: () => {
        this.addingTag = false
      }
    })
    return true
  }


}
